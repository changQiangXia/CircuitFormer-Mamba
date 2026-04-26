import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _ensure_finite(stage, tensor):
    if torch.isfinite(tensor).all():
        return
    finite_mask = torch.isfinite(tensor)
    nonfinite_count = int((~finite_mask).sum().item())
    if finite_mask.any():
        finite_values = tensor[finite_mask]
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_mean = float(finite_values.mean().item())
    else:
        finite_min = float('nan')
        finite_max = float('nan')
        finite_mean = float('nan')
    raise FloatingPointError(
        f'TrueMambaBlock produced non-finite activations at stage={stage}: '
        f'shape={tuple(tensor.shape)} dtype={tensor.dtype} '
        f'nonfinite_count={nonfinite_count} '
        f'finite_min={finite_min} finite_max={finite_max} '
        f'finite_mean={finite_mean}'
    )


class TrueMambaBlock(nn.Module):
    """
    Applies the official 1D Mamba block to row and column sequences
    from a 2D BEV feature map.
    """

    def __init__(
        self,
        dim=64,
        d_state=16,
        d_conv=4,
        expand=2,
        downsample=1,
        bidirectional=True,
        use_input_norm=True,
        use_mask=False,
        mask_pool_mode="max",
        out_proj_init_zero=True,
        out_proj_init_std=0.0,
        use_residual_scale=False,
        residual_scale_init=1.0,
        remask_after_upsample=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.downsample = max(1, int(downsample))
        self.bidirectional = bool(bidirectional)
        self.use_input_norm = bool(use_input_norm)
        self.use_mask = bool(use_mask)
        self.mask_pool_mode = str(mask_pool_mode)
        self.remask_after_upsample = bool(remask_after_upsample)

        self.norm = nn.GroupNorm(1, self.dim) if self.use_input_norm else nn.Identity()
        self.row_mamba = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.col_mamba = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.out_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)
        if out_proj_init_zero:
            nn.init.zeros_(self.out_proj.weight)
        elif float(out_proj_init_std) > 0:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(out_proj_init_std))
        self.residual_scale = None
        if use_residual_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    @staticmethod
    def _build_occupancy_mask(x):
        return x.abs().sum(dim=1, keepdim=True).gt(0).to(dtype=x.dtype)

    def _pool_mask(self, mask):
        if self.downsample <= 1:
            return mask
        if self.mask_pool_mode == "max":
            pooled = F.max_pool2d(mask, kernel_size=self.downsample, stride=self.downsample)
        elif self.mask_pool_mode == "avg":
            pooled = F.avg_pool2d(mask, kernel_size=self.downsample, stride=self.downsample)
        else:
            raise ValueError(f"Unsupported mask_pool_mode: {self.mask_pool_mode}")
        return pooled.gt(0).to(dtype=mask.dtype)

    def _apply_sequence_mixer(self, mixer, seq):
        forward = mixer(seq)
        if not self.bidirectional:
            return forward
        backward = torch.flip(mixer(torch.flip(seq, dims=[1])), dims=[1])
        return 0.5 * (forward + backward)

    def _scan_rows(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)
        seq = self._apply_sequence_mixer(self.row_mamba, seq)
        return seq.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def _scan_cols(self, x):
        batch_size, channels, height, width = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)
        seq = self._apply_sequence_mixer(self.col_mamba, seq)
        return seq.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1)

    def forward(self, x):
        residual = x
        # Official Mamba is numerically fragile under fp16 autocast on this path,
        # so the state-space scan is kept in fp32 and cast back afterwards.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = x.float()
            full_mask = self._build_occupancy_mask(y) if self.use_mask else None
            y = self.norm(y)
            _ensure_finite('norm', y)
            if full_mask is not None:
                y = y * full_mask
                _ensure_finite('input_mask', y)

            scan_mask = full_mask
            if self.downsample > 1:
                y = F.avg_pool2d(y, kernel_size=self.downsample, stride=self.downsample)
                _ensure_finite('downsample', y)
                if scan_mask is not None:
                    scan_mask = self._pool_mask(scan_mask)
                    y = y * scan_mask
                    _ensure_finite('downsample_mask', y)

            row_y = self._scan_rows(y)
            col_y = self._scan_cols(y)
            _ensure_finite('row_scan', row_y)
            _ensure_finite('col_scan', col_y)
            y = 0.5 * (row_y + col_y)
            _ensure_finite('scan_mix', y)
            if scan_mask is not None:
                y = y * scan_mask
                _ensure_finite('scan_remask', y)
            y = self.out_proj(y)
            _ensure_finite('out_proj', y)
            if scan_mask is not None:
                y = y * scan_mask
                _ensure_finite('out_proj_remask', y)

            if self.downsample > 1:
                y = F.interpolate(y, size=residual.shape[-2:], mode="bilinear", align_corners=False)
                _ensure_finite('upsample', y)
                if self.remask_after_upsample and full_mask is not None:
                    y = y * full_mask
                    _ensure_finite('upsample_remask', y)
            if self.residual_scale is not None:
                y = y * self.residual_scale
                _ensure_finite('residual_scale', y)
        y = y.to(dtype=residual.dtype)
        _ensure_finite('cast_back', y)
        output = residual + y
        _ensure_finite('residual_add', output)
        return output


class TrueMambaNeck(nn.Module):
    def __init__(
        self,
        dim=64,
        num_blocks=1,
        d_state=16,
        d_conv=4,
        expand=2,
        downsample=1,
        bidirectional=True,
        use_input_norm=True,
        use_mask=False,
        mask_pool_mode="max",
        out_proj_init_zero=True,
        out_proj_init_std=0.0,
        use_residual_scale=False,
        residual_scale_init=1.0,
        remask_after_upsample=True,
    ):
        super().__init__()
        blocks = []
        for _ in range(int(num_blocks)):
            blocks.append(
                TrueMambaBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    downsample=downsample,
                    bidirectional=bidirectional,
                    use_input_norm=use_input_norm,
                    use_mask=use_mask,
                    mask_pool_mode=mask_pool_mode,
                    out_proj_init_zero=out_proj_init_zero,
                    out_proj_init_std=out_proj_init_std,
                    use_residual_scale=use_residual_scale,
                    residual_scale_init=residual_scale_init,
                    remask_after_upsample=remask_after_upsample,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


TrueMambaNeckPrototype = TrueMambaNeck


def build_true_mamba_neck(model_cfg, dim):
    true_cfg = _cfg_get(model_cfg, "true_mamba", None)
    if not _cfg_get(true_cfg, "enabled", False):
        return nn.Identity()
    return TrueMambaNeck(
        dim=dim,
        num_blocks=_cfg_get(true_cfg, "num_blocks", 1),
        d_state=_cfg_get(true_cfg, "d_state", 16),
        d_conv=_cfg_get(true_cfg, "d_conv", 4),
        expand=_cfg_get(true_cfg, "expand", 2),
        downsample=_cfg_get(true_cfg, "downsample", 1),
        bidirectional=_cfg_get(true_cfg, "bidirectional", True),
        use_input_norm=_cfg_get(true_cfg, "use_input_norm", True),
        use_mask=_cfg_get(true_cfg, "use_mask", False),
        mask_pool_mode=_cfg_get(true_cfg, "mask_pool_mode", "max"),
        out_proj_init_zero=_cfg_get(true_cfg, "out_proj_init_zero", True),
        out_proj_init_std=_cfg_get(true_cfg, "out_proj_init_std", 0.0),
        use_residual_scale=_cfg_get(true_cfg, "use_residual_scale", False),
        residual_scale_init=_cfg_get(true_cfg, "residual_scale_init", 1.0),
        remask_after_upsample=_cfg_get(true_cfg, "remask_after_upsample", True),
    )
