"""Parameter-matched additive refinement heads for the BUS-BRA comparison."""

import torch
import torch.nn.functional as F
from torch import nn

from .scbt_net import ZeroInitResidualHead, _validate_like


class _BoundaryAttention(nn.Module):
    def forward(self, hidden, probability):
        return hidden * (1 - (2 * probability - 1).abs()).to(hidden.dtype)


class _FourierGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(1, channels, 3, 3))

    def forward(self, hidden, probability):
        height, width = hidden.shape[-2:]
        with torch.autocast(hidden.device.type, enabled=False):
            rows = torch.arange(height, device=hidden.device)
            even_rows = torch.minimum(rows, height - rows)
            gain = F.interpolate(self.log_gain.float(), (height // 2 + 1, width // 2 + 1),
                                 mode="bilinear", align_corners=True).index_select(2, even_rows)
            spectrum = torch.fft.rfft2(hidden.float().contiguous(), norm="ortho")
            filtered = torch.fft.irfft2(spectrum * (2 * gain.sigmoid()),
                                        s=(height, width), norm="ortho")
        return filtered.to(hidden.dtype).contiguous(memory_format=torch.channels_last)


class _HaarGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        basis = 0.5 * torch.tensor([
            [[1., 1.], [1., 1.]], [[1., 1.], [-1., -1.]],
            [[1., -1.], [1., -1.]], [[1., -1.], [-1., 1.]],
        ])
        self.register_buffer("filters", basis[:, None].repeat(channels, 1, 1, 1))
        self.log_gain = nn.Parameter(torch.zeros(1, 4 * channels, 1, 1))

    def forward(self, hidden, probability):
        if any(size % 2 for size in hidden.shape[-2:]):
            raise ValueError("Haar refinement requires even height and width.")
        with torch.autocast(hidden.device.type, enabled=False):
            filters = self.filters.float()
            bands = F.conv2d(hidden.float(), filters, stride=2, groups=self.channels)
            gated = bands * (2 * self.log_gain.float().sigmoid())
            restored = F.conv_transpose2d(gated, filters, stride=2, groups=self.channels)
        return restored.to(hidden.dtype)


class PlainCascadeRefiner(nn.Module):
    """G1 ordinary cascade with 16,073 trainable parameters."""

    def __init__(self, transform=None, width=98):
        super().__init__()
        self.transform = transform
        self.region = ZeroInitResidualHead(52, width)

    def forward(self, hidden, logits):
        if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
            raise ValueError("Expected floating-point coarse logits of shape [B, 1, H, W].")
        _validate_like(hidden, logits, 50, "Hidden feature")
        probability = logits.sigmoid()
        if self.transform is not None:
            hidden = self.transform(hidden, probability)
        return logits + self.region(torch.cat((hidden, logits, probability), dim=1))


class ReverseAttentionRefiner(PlainCascadeRefiner):
    """R-Attn with 16,073 trainable parameters."""

    def __init__(self):
        super().__init__(_BoundaryAttention())


class FFTSpectralRefiner(PlainCascadeRefiner):
    """R-FFT with 16,262 trainable parameters."""

    def __init__(self):
        super().__init__(_FourierGate(50), width=97)


class HaarWaveletRefiner(PlainCascadeRefiner):
    """R-Wavelet with 16,273 trainable parameters."""

    def __init__(self):
        super().__init__(_HaarGate(50))
