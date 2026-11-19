"""Contour-aligned spectral refinement for binary segmentation."""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


def _group_norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


def _project(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        _group_norm(out_channels),
        nn.SiLU(),
    )


class _SpatialResidual(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            _group_norm(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.body(x)


def _gaussian_kernel(sigma):
    radius = math.ceil(3 * sigma)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * (coords / sigma).square())
    return weights / weights.sum()


def _blur(x, kernel, mode):
    channels = x.shape[1]
    radius = kernel.numel() // 2
    if mode == "reflect" and min(x.shape[-2:]) <= radius:
        raise ValueError(f"Gaussian reflection requires each spatial axis to exceed {radius}.")
    row = kernel.to(x.dtype).reshape(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
    blurred = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode=mode), row, groups=channels)
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode=mode),
                    row.transpose(-1, -2).contiguous(), groups=channels)


def _validate_like(value, logits, channels, name):
    expected = (logits.shape[0], channels, *logits.shape[-2:])
    if value.shape != expected or value.device != logits.device or not value.is_floating_point():
        raise ValueError(f"{name} must be floating-point with shape {expected} on {logits.device}.")


class GaussianSpectralBands(nn.Module):
    """Fixed separable Gaussians. Return (low, mid, high), each shaped like the input."""

    def __init__(self, short_sigma=1.5, long_sigma=3.0):
        super().__init__()
        if not 0 < short_sigma < long_sigma:
            raise ValueError("Expected 0 < short_sigma < long_sigma.")
        self.register_buffer("short_kernel", _gaussian_kernel(short_sigma), persistent=False)
        self.register_buffer("long_kernel", _gaussian_kernel(long_sigma), persistent=False)

    def forward(self, x):
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError("Gaussian bands require a floating-point [B, C, H, W] tensor.")
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            short = _blur(x, self.short_kernel, "reflect")
            low = _blur(x, self.long_kernel, "reflect")
            return low, short - low, x - short


class ContourAlignedDirectionalDerivatives(nn.Module):
    """Project three aligned 32-channel bands into a 32-channel evidence tensor."""

    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(_project(32, 16) for _ in range(3))
        self.fuse = nn.Sequential(*_project(132, 32), _SpatialResidual(32))
        self.register_buffer("normal_kernel", _gaussian_kernel(2.0), persistent=False)
        x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("sobel_kernel", torch.stack((x, x.T))[:, None], persistent=False)

    def _gradient(self, x):
        batch, channels, height, width = x.shape
        with torch.autocast(x.device.type, enabled=False):
            kernels = self.sobel_kernel.float().repeat(channels, 1, 1, 1)
            gradients = F.conv2d(F.pad(x.float(), (1, 1, 1, 1), mode="replicate"),
                                 kernels, groups=channels)
        gradients = gradients.reshape(batch, channels, 2, height, width)
        return gradients[:, :, 0], gradients[:, :, 1]

    def geometry(self, logits):
        """Return the reliability-weighted normal, gradient magnitude and soft band."""
        with torch.autocast(logits.device.type, enabled=False):
            smoothed = _blur(logits.float(), self.normal_kernel, "replicate")
            gx, gy = self._gradient(smoothed)
            magnitude = (gx.square() + gy.square() + 1e-4).sqrt()
            normal = torch.cat((gx, gy), dim=1) * (magnitude / 0.5).tanh() / magnitude
            soft_band = (-smoothed.square() / (2 * 3.0 ** 2)).exp()
        return normal, magnitude, soft_band.detach()

    def forward(self, bands, logits):
        if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
            raise ValueError("Expected floating-point coarse logits of shape [B, 1, H, W].")
        if len(bands) != 3:
            raise ValueError("Expected low, mid and high Gaussian bands in that order.")
        for band in bands:
            _validate_like(band, logits, 32, "Gaussian band")
        normal, magnitude, soft_band = self.geometry(logits)
        features = torch.cat([projection(band) for projection, band in zip(self.project, bands)], dim=1)
        gx, gy = self._gradient(features)
        nx, ny = normal.split(1, dim=1)
        normal_response = nx * gx + ny * gy
        high_tangent = (-ny * gx + nx * gy).chunk(3, dim=1)[2]
        low_normal, _, high_normal = normal_response.chunk(3, dim=1)
        fused = torch.cat((features, normal_response, high_tangent, high_normal - low_normal,
                           logits, logits.sigmoid(), magnitude, soft_band), dim=1)
        return self.fuse(fused)


class ZeroInitResidualHead(nn.Module):
    """Pointwise projection, dilated depthwise residual block and zero-init logit."""

    def __init__(self, in_channels=85, width=57):
        super().__init__()
        self.enter = _project(in_channels, width)
        self.blocks = _SpatialResidual(width)
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.blocks(self.enter(x))).float()


class SCBTNetRefiner(nn.Module):
    """The 16,189-parameter B1 head. Inputs and output use the coarse logit grid."""

    def __init__(self):
        super().__init__()
        self.gaussian = GaussianSpectralBands()
        self.evidence = ContourAlignedDirectionalDerivatives()
        self.region = ZeroInitResidualHead()

    def forward(self, hidden, logits, *, band_source=None, bands=None):
        if logits.ndim != 4 or logits.shape[1] != 1 or not logits.is_floating_point():
            raise ValueError("Expected floating-point coarse logits of shape [B, 1, H, W].")
        _validate_like(hidden, logits, 50, "Hidden feature")
        if (band_source is None) == (bands is None):
            raise ValueError("Provide exactly one of band_source or precomputed bands.")
        if bands is None:
            _validate_like(band_source, logits, 32, "Band source")
            bands = self.gaussian(band_source)
        evidence = self.evidence(bands, logits)
        # The B1 region tensor retains both coarse and transport slots. B1 uses
        # the coarse tensor in both slots since it has no spatial transport.
        residual = self.region(torch.cat((hidden, evidence, logits, logits, logits.sigmoid()), dim=1))
        return logits + residual

    @classmethod
    def attach(cls, backbone):
        """Freeze a backbone returning (logits, three feature levels) or a native mapping."""
        return UniversalSCBTAdapter(backbone, cls())


class UniversalSCBTAdapter(nn.Module):
    """Adapt a binary segmenter's features without adding trainable adapter parameters.

    The wrapped parent returns (logits, levels) or a mapping with logits, hidden,
    and optionally native bands. Feature levels are [B, C_i, H_i, W_i] tensors.
    """

    def __init__(self, backbone, refiner=None):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.refiner = SCBTNetRefiner() if refiner is None else refiner

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _pool_channels(feature, count):
        return F.adaptive_avg_pool3d(
            feature.unsqueeze(1), (count, *feature.shape[-2:])
        ).squeeze(1)

    def forward(self, x):
        with torch.no_grad():
            coarse = self.backbone(x)
            if isinstance(coarse, Mapping):
                logits = coarse["logits"]
                hidden, bands = coarse["hidden"], coarse.get("bands")
            else:
                logits, levels = coarse
                if len(levels) != 3:
                    raise ValueError("The backbone adapter requires exactly three feature levels.")
                with torch.autocast(logits.device.type, enabled=False):
                    aligned = []
                    for feature, channels in zip(levels, (16, 17, 17)):
                        if (feature.ndim != 4 or feature.shape[0] != logits.shape[0]
                                or feature.device != logits.device or not feature.is_floating_point()):
                            raise ValueError("Invalid feature level shape, device or dtype.")
                        aligned.append(F.interpolate(
                            self._pool_channels(feature.float(), channels), size=logits.shape[-2:],
                            mode="bilinear", align_corners=False,
                        ))
                    hidden = torch.cat(aligned, dim=1)
                bands = None
            if isinstance(self.refiner, SCBTNetRefiner) and bands is None:
                with torch.autocast(logits.device.type, enabled=False):
                    source = self._pool_channels(hidden.float(), 32)
            else:
                source = None
        if isinstance(self.refiner, SCBTNetRefiner):
            return self.refiner(hidden, logits, band_source=source, bands=bands)
        return self.refiner(hidden, logits)
