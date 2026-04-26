# True Mamba Environment

`true Mamba Scheme B` 使用独立环境承载。当前服务器上完成验证的环境名为 `circuitformer-true-mamba`。

## 当前验证组合

- Python: `3.10.20`
- Torch: `2.5.1+cu121`
- TorchVision: `0.20.1+cu121`
- TorchAudio: `2.5.1+cu121`
- `spconv-cu121==2.3.8`
- `torch-scatter==2.1.2+pt25cu121`
- `causal-conv1d==1.6.1`
- `mamba-ssm==2.3.1`

## 依赖文件

- 通用 Python 依赖: `true_mamba_experiments/requirements-true-mamba.txt`
- 环境创建脚本: `true_mamba_experiments/scripts/create_true_mamba_env.sh`
- 环境验证脚本: `true_mamba_experiments/scripts/verify_true_mamba_env.sh`

## 额外固定项

- `numpy==1.23.5`
  原因: 当前工程中的 `metrics.py` 仍使用 `np.float`
- `setuptools==75.8.0`
  原因: `pytorch-lightning 2.1.0` 的导入路径仍会访问 `pkg_resources`
- `wandb==0.13.5`
  原因: 当前训练入口默认创建 `WandbLogger`

## 环境落地过程中的关键信号

- 大体积 wheel 安装阶段需要配合 `source /etc/network_turbo || true` 与 `--no-cache-dir`
- 磁盘压力集中出现在 `/tmp/pip-build-env-*`、`/tmp/pip-unpack-*` 与 `/root/.cache/pip`
- 宿主机曾出现 `OMP_NUM_THREADS=0`，后续脚本统一显式设置 `OMP_NUM_THREADS=1`

## 已完成验证

- 官方 `Mamba` CUDA 前向通过
  输入形状: `[2, 32, 64]`
  输出形状: `[2, 32, 64]`
- 仓库自带 `scripts/smoke_test_cpu.py` 通过
- `true_mamba_experiments/scripts/verify_true_mamba_env.sh` 已形成单命令验证闭环

## 推荐流程

1. `bash true_mamba_experiments/scripts/create_true_mamba_env.sh`
2. `bash true_mamba_experiments/scripts/verify_true_mamba_env.sh`
3. `bash true_mamba_experiments/scripts/preflight_true_mamba_scheme_b.sh`
4. `bash true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh`
