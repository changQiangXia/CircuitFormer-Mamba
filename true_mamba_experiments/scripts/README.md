# Script Notes

本目录保存 true Mamba 路线的环境、preflight、训练与诊断脚本。

## 环境与验证

- `create_true_mamba_env.sh`
- `verify_true_mamba_env.sh`

## 冒烟与诊断

- `smoke_shape_audit.py`
- `smoke_true_mamba_neck_module.py`
- `smoke_true_mamba_full_model.py`
- `smoke_true_mamba_one_step.py`
- `diagnose_true_mamba_nan_ckpt.py`

## preflight 与训练

- `preflight_true_mamba_train_entry.sh`
- `preflight_true_mamba_scheme_b.sh`
- `train_congestion_true_mamba_run1.sh`
- `train_congestion_true_mamba_run2_allocator_guard.sh`
- `train_congestion_true_mamba_run3_bf16_guard.sh`
- `train_congestion_true_mamba_scheme_b.sh`
- `resume_congestion_true_mamba_run1.sh`
- `monitor_true_mamba_run.sh`

## 当前建议入口

1. `bash true_mamba_experiments/scripts/create_true_mamba_env.sh`
2. `bash true_mamba_experiments/scripts/verify_true_mamba_env.sh`
3. `bash true_mamba_experiments/scripts/preflight_true_mamba_scheme_b.sh`
4. `bash true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh`
