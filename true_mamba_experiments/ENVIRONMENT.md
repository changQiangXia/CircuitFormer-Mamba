# True Mamba Environment

`true Mamba Scheme B` 使用独立环境承载。当前服务器上完成验证的环境名为 `circuitformer-true-mamba`。

## 与主线环境的关系

仓库内当前长期保留两套环境:

- `circuitformer-gpu`
  - 面向原版 `CircuitFormer`、强基线与 `zero-init BEV Mamba`
  - 已验证组合为 `Python 3.9.25 + torch 1.13.0+cu117`
- `circuitformer-true-mamba`
  - 面向 `true Mamba Scheme B`
  - 已验证组合为 `Python 3.10.20 + torch 2.5.1+cu121`

拆分原因可归纳为三层:

- 第一层是 Torch 栈不同，随之变化的还有 `spconv` 与 `torch-scatter` 的安装入口。
- 第二层是 official `mamba-ssm` 的已验证组合位于 `torch 2.5.1+cu121`，主线归档组合位于 `torch 1.13.0+cu117`。
- 第三层是 true Mamba 训练阶段还附带 `bf16-mixed`、finite guard、mask、额外初始化约束等稳定性处理，排障路径与主线环境不同。

因此，`circuitformer-true-mamba` 更适合视为“official-Mamba 独立实验环境”，`circuitformer-gpu` 更适合视为“主线复现环境”。

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

## 配置流程

从零开始时，执行顺序建议固定为四步:

1. 创建环境

```bash
bash true_mamba_experiments/scripts/create_true_mamba_env.sh
```

2. 激活环境

```bash
conda activate circuitformer-true-mamba
```

3. 验证环境

```bash
bash true_mamba_experiments/scripts/verify_true_mamba_env.sh
```

4. 进入 preflight 或正式训练

```bash
bash true_mamba_experiments/scripts/preflight_true_mamba_scheme_b.sh
bash true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh
```

这四步分别对应:

- 安装依赖
- 切换到 official-Mamba 环境
- 验证 `Mamba` CUDA 前向与仓库主干冒烟测试
- 启动 Scheme B 的预检查与正式训练

## 额外固定项

- `numpy==1.23.5`
  原因: 当前工程中的 `metrics.py` 仍使用 `np.float`
- `setuptools==75.8.0`
  原因: `pytorch-lightning 2.1.0` 的导入路径仍会访问 `pkg_resources`
- `wandb==0.13.5`
  原因: 当前训练入口默认创建 `WandbLogger`

## 环境落地过程中的关键信号

- `source /etc/network_turbo || true` 是 AutoDL 平台提供的网络加速入口；其他平台或本地机器通常没有这一脚本，相关依赖下载需自行处理网络连接
- 大体积 wheel 安装阶段可在 AutoDL 平台上配合 `source /etc/network_turbo || true` 与 `--no-cache-dir`
- 磁盘压力集中出现在 `/tmp/pip-build-env-*`、`/tmp/pip-unpack-*` 与 `/root/.cache/pip`
- 宿主机曾出现 `OMP_NUM_THREADS=0`，后续脚本统一显式设置 `OMP_NUM_THREADS=1`

## 已完成验证

- 官方 `Mamba` CUDA 前向通过
  输入形状: `[2, 32, 64]`
  输出形状: `[2, 32, 64]`
- 仓库自带 `scripts/smoke_test_cpu.py` 通过
- `true_mamba_experiments/scripts/verify_true_mamba_env.sh` 已形成单命令验证闭环

若验证阶段失败，可按故障位置区分处理:

- `Mamba` CUDA 前向失败
  重点回看 Torch、CUDA、`causal-conv1d`、`mamba-ssm`
- 仓库主干冒烟失败
  重点回看 `spconv`、`torch-scatter`、`numpy`、`setuptools`
- preflight 失败
  重点回看训练入口参数、数据路径、精度设置与显存状态

## 推荐流程

1. `bash true_mamba_experiments/scripts/create_true_mamba_env.sh`
2. `bash true_mamba_experiments/scripts/verify_true_mamba_env.sh`
3. `bash true_mamba_experiments/scripts/preflight_true_mamba_scheme_b.sh`
4. `bash true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh`
