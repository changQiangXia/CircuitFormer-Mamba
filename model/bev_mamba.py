import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SequenceStateSpace(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.delta_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, seq):
        delta = torch.sigmoid(self.delta_proj(seq))
        value = self.value_proj(seq)
        state = torch.zeros(seq.shape[0], seq.shape[-1], device=seq.device, dtype=seq.dtype)
        outputs = []
        for idx in range(seq.shape[1]):
            state = (1.0 - delta[:, idx, :]) * state + delta[:, idx, :] * value[:, idx, :]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class LightweightBEVMambaBlock(nn.Module):
    def __init__(self, dim, inner_dim, scan_downsample, dw_kernel_size, out_proj_init_zero=False):
        super().__init__()
        self.scan_downsample = max(1, int(scan_downsample))
        self.norm = nn.GroupNorm(1, dim)
        self.in_proj = nn.Conv2d(dim, inner_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            inner_dim,
            inner_dim,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=inner_dim,
            bias=False,
        )
        self.row_scan = SequenceStateSpace(inner_dim)
        self.col_scan = SequenceStateSpace(inner_dim)
        self.out_proj = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=False)
        if out_proj_init_zero:
            nn.init.zeros_(self.out_proj.weight)

    def _scan_rows(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)
        forward = self.row_scan(seq)
        backward = torch.flip(self.row_scan(torch.flip(seq, dims=[1])), dims=[1])
        out = 0.5 * (forward + backward)
        return out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def _scan_cols(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)
        forward = self.col_scan(seq)
        backward = torch.flip(self.col_scan(torch.flip(seq, dims=[1])), dims=[1])
        out = 0.5 * (forward + backward)
        return out.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        if self.scan_downsample > 1:
            y = F.avg_pool2d(y, kernel_size=self.scan_downsample, stride=self.scan_downsample)

        y, gate = self.in_proj(y).chunk(2, dim=1)
        y = F.silu(self.dwconv(y))
        y = 0.5 * (self._scan_rows(y) + self._scan_cols(y))
        y = y * torch.sigmoid(gate)
        y = self.out_proj(y)

        if self.scan_downsample > 1:
            y = F.interpolate(y, size=residual.shape[-2:], mode='bilinear', align_corners=False)
        return residual + y


class BEVMambaNeck(nn.Module):
    def __init__(self, dim, num_blocks=1, inner_dim=None, scan_downsample=4, dw_kernel_size=3, out_proj_init_zero=False):
        super().__init__()
        inner_dim = dim if inner_dim is None else int(inner_dim)
        blocks = []
        for _ in range(int(num_blocks)):
            blocks.append(
                LightweightBEVMambaBlock(
                    dim=dim,
                    inner_dim=inner_dim,
                    scan_downsample=scan_downsample,
                    dw_kernel_size=dw_kernel_size,
                    out_proj_init_zero=out_proj_init_zero,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def build_bev_mamba_neck(model_cfg, dim):
    bev_cfg = _cfg_get(model_cfg, 'bev_mamba', None)
    if not _cfg_get(bev_cfg, 'enabled', False):
        return nn.Identity()
    return BEVMambaNeck(
        dim=dim,
        num_blocks=_cfg_get(bev_cfg, 'num_blocks', 1),
        inner_dim=_cfg_get(bev_cfg, 'inner_dim', dim),
        scan_downsample=_cfg_get(bev_cfg, 'scan_downsample', 4),
        dw_kernel_size=_cfg_get(bev_cfg, 'dw_kernel_size', 3),
        out_proj_init_zero=_cfg_get(bev_cfg, 'out_proj_init_zero', False),
    )
