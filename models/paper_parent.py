"""Reference BUSI U-Net/G2-V2 interface for externally supplied checkpoints."""

from collections.abc import Mapping

import torch
from torch import nn

from .scbt_net import GaussianSpectralBands, ZeroInitResidualHead, _SpatialResidual, _blur, _project


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _UNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        for index, channels in enumerate((32, 64, 128, 256), 1):
            setattr(self, f"enc{index}", _DoubleConv(3 if index == 1 else channels // 2, channels))
            setattr(self, f"up{index}", nn.ConvTranspose2d(2 * channels, channels, 2, stride=2))
            setattr(self, f"dec{index}", _DoubleConv(2 * channels, channels))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(256, 512)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        if any(size % 16 for size in x.shape[-2:]):
            raise ValueError("The paper U-Net requires both input axes to be divisible by 16.")
        skips = []
        for index in range(1, 5):
            x = getattr(self, f"enc{index}")(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for index in range(4, 0, -1):
            x = getattr(self, f"up{index}")(x)
            x = getattr(self, f"dec{index}")(torch.cat((x, skips[index - 1]), dim=1))
        return skips[0], x, self.out_conv(x)


class _BaselineBands(nn.Module):
    def __init__(self):
        super().__init__()
        gaussian = GaussianSpectralBands()
        self.register_buffer("k0", gaussian.short_kernel)
        self.register_buffer("k1", gaussian.long_kernel)
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(32, 2), nn.SiLU(), nn.Linear(2, 32)) for _ in range(3)
        )
        self.merge = nn.Conv2d(96, 32, 1, groups=32)

    def forward(self, x):
        # Use checkpoint kernels and the parent model's current precision.
        short = _blur(x, self.k0, "reflect")
        low = _blur(x, self.k1, "reflect")
        bands = (low, short - low, x - short)
        gates = [2 * gate(band.mean((2, 3))).sigmoid() for gate, band in zip(self.gates, bands)]
        weighted = [band * gain[:, :, None, None] for band, gain in zip(bands, gates)]
        return bands, self.merge(torch.stack(weighted, dim=2).flatten(1, 2))


class _G2V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = _project(65, 32)
        self.bands = _BaselineBands()
        self.unidirectional = ZeroInitResidualHead(32, 50)
        self.unidirectional.blocks = nn.Sequential(
            *(_SpatialResidual(50, dilation) for dilation in (1, 2, 4))
        )

    def forward(self, skip, decoder, logits):
        bands, merged = self.bands(self.project(torch.cat((skip, decoder, logits), dim=1)))
        region = self.unidirectional
        hidden = region.blocks(region.enter(merged))
        return {"logits": logits + region.out(hidden).float(), "hidden": hidden, "bands": bands}


class PaperParent(nn.Module):
    """Binary B0 with native feature tensors for the paper's ultrasound comparison."""

    def __init__(self):
        super().__init__()
        self.backbone = _UNetBackbone()
        self.head = _G2V2()

    def forward(self, x):
        return self.head(*self.backbone(x))


def load_parent(*, fold, weights_dir):
    """Load the shared BUSI parent and the fold-specific G2-V2 head strictly."""
    if weights_dir is None:
        raise ValueError("--parent-weights must point to the directory of external B0 weights.")
    unet = torch.load(weights_dir / "busi_unet.pth", map_location="cpu", weights_only=True)
    head = torch.load(weights_dir / f"fold_{fold}_g2v2.pth", map_location="cpu", weights_only=True)
    if isinstance(unet, Mapping) and "model" in unet:
        unet = unet["model"]
    if isinstance(head, Mapping) and "arms" in head:
        head = head["arms"]["B0"]["state"]
    elif isinstance(head, Mapping) and "ema" in head:
        head = head["ema"]
    model = PaperParent()
    model.backbone.load_state_dict({
        key.removeprefix("backbone."): value for key, value in unet.items()
    }, strict=True)
    model.head.load_state_dict(head, strict=True)
    return model.requires_grad_(False).eval()
