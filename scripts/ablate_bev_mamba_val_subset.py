#!/usr/bin/env python
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.circuitnet import Circuitnet
from data.data_interface import collate_fn
from model.circuitformer import CircuitFormer


def _default_config():
    cfg = OmegaConf.load(REPO_ROOT / 'config' / 'config.yaml')
    cfg.data.data_root = str(REPO_ROOT.parent / 'datasets' / 'CircuitNet-N28' / 'graph_features' / 'instance_placement_micron')
    cfg.data.label_root = str(REPO_ROOT.parent / 'datasets' / 'CircuitNet-N28' / 'training_set' / 'congestion' / 'label')
    return cfg


def _make_input(batch):
    return [batch[0], batch[1], batch[2], batch[3], batch[4]]


def _pearson(pred, target):
    a = pred.reshape(-1).float()
    b = target.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.square().mean().sqrt() * b.square().mean().sqrt()).item()
    if denom == 0:
        return 0.0
    return ((a * b).mean().item()) / denom


def _mse(pred, target):
    return torch.mean((pred.float() - target.float()) ** 2).item()


def _tv(x):
    dx = (x[..., 1:] - x[..., :-1]).abs().mean().item()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean().item()
    return 0.5 * (dx + dy)


def _load_model(cfg, ckpt_path, *, mamba_enabled, scan_downsample, bypass_neck=False, zero_out_proj=False):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.model.bev_mamba.enabled = mamba_enabled
    cfg.model.bev_mamba.num_blocks = 1
    cfg.model.bev_mamba.inner_dim = 64
    cfg.model.bev_mamba.scan_downsample = scan_downsample
    cfg.model.bev_mamba.dw_kernel_size = 3
    cfg.model.bev_mamba.out_proj_init_zero = False

    model = CircuitFormer(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {
        key[len('model.'):]: value
        for key, value in ckpt['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(state_dict, strict=False)

    if bypass_neck:
        model.bev_neck = torch.nn.Identity()
    elif zero_out_proj and hasattr(model.bev_neck, 'blocks'):
        for block in model.bev_neck.blocks:
            if hasattr(block, 'out_proj'):
                torch.nn.init.zeros_(block.out_proj.weight)

    model.eval()
    return model


def _variant_specs(args):
    return {
        'baseline_best': {
            'ckpt_path': args.baseline_ckpt,
            'mamba_enabled': False,
            'scan_downsample': 4,
            'bypass_neck': False,
            'zero_out_proj': False,
        },
        'mamba_ds4': {
            'ckpt_path': args.mamba_ckpt,
            'mamba_enabled': True,
            'scan_downsample': 4,
            'bypass_neck': False,
            'zero_out_proj': False,
        },
        'mamba_ds1': {
            'ckpt_path': args.mamba_ckpt,
            'mamba_enabled': True,
            'scan_downsample': 1,
            'bypass_neck': False,
            'zero_out_proj': False,
        },
        'mamba_identity': {
            'ckpt_path': args.mamba_ckpt,
            'mamba_enabled': True,
            'scan_downsample': 4,
            'bypass_neck': True,
            'zero_out_proj': False,
        },
        'mamba_zero_outproj': {
            'ckpt_path': args.mamba_ckpt,
            'mamba_enabled': True,
            'scan_downsample': 4,
            'bypass_neck': False,
            'zero_out_proj': True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=16)
    parser.add_argument(
        '--variants',
        nargs='+',
        default=['baseline_best', 'mamba_ds4', 'mamba_ds1', 'mamba_identity', 'mamba_zero_outproj'],
    )
    parser.add_argument(
        '--baseline-ckpt',
        default=str(REPO_ROOT / 'exp' / 'congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC' / 'epoch=99-pearson=0.6488.ckpt'),
    )
    parser.add_argument(
        '--mamba-ckpt',
        default=str(REPO_ROOT / 'exp' / 'congestion_bev_mamba_run1_2026-04-13_10-01-21_UTC' / 'epoch=19-pearson=0.5430.ckpt'),
    )
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    args = parser.parse_args()

    cfg = _default_config()
    dataset = Circuitnet(split={'split': args.split}, data_root=cfg.data.data_root, label_root=cfg.data.label_root)
    specs = _variant_specs(args)
    models = {
        name: _load_model(cfg, **specs[name])
        for name in args.variants
    }

    results = {}
    with torch.no_grad():
        for name, model in models.items():
            metrics = {
                'pearson': [],
                'mse': [],
                'pred_tv': [],
                'pred_std': [],
            }
            if getattr(model, 'bev_neck', None) is not None and not isinstance(model.bev_neck, torch.nn.Identity):
                metrics['feat_delta'] = []
                metrics['feat_tv_ratio'] = []

            for idx in range(args.subset):
                batch = collate_fn([dataset[idx]])
                label = batch[5]
                encoded = model.encoder(_make_input(batch))
                feat = model.bev_neck(encoded)
                pred = model.decoder(feat)

                metrics['pearson'].append(_pearson(pred, label))
                metrics['mse'].append(_mse(pred, label))
                metrics['pred_tv'].append(_tv(pred))
                metrics['pred_std'].append(pred.std().item())

                if 'feat_delta' in metrics:
                    encoded_norm = encoded.norm().item() + 1e-12
                    encoded_tv = _tv(encoded) + 1e-12
                    metrics['feat_delta'].append((feat - encoded).norm().item() / encoded_norm)
                    metrics['feat_tv_ratio'].append(_tv(feat) / encoded_tv)

            results[name] = {
                metric: statistics.mean(values)
                for metric, values in metrics.items()
            }

    print(json.dumps(
        {
            'subset': args.subset,
            'split': args.split,
            'results': results,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == '__main__':
    main()
