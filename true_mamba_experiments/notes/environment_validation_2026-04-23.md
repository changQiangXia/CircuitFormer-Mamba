# True Mamba Environment Validation 2026-04-23

## 目标

为真正 Mamba 实验准备与主线归档环境隔离的独立运行环境，并验证两件事:

- 官方 `mamba-ssm` 能否在当前机器上完成 CUDA 前向
- 当前 CircuitFormer 工程能否在该环境下完成基础级冒烟验证

## 环境结论

- 环境名: `circuitformer-true-mamba`
- Python: `3.10`
- Torch: `2.5.1+cu121`
- TorchVision: `0.20.1+cu121`
- TorchAudio: `2.5.1+cu121`
- GPU: `NVIDIA GeForce RTX 3090`
- 官方 Mamba 依赖:
  - `causal-conv1d==1.6.1`
  - `mamba-ssm==2.3.1`

## 额外固定项

- `numpy==1.23.5`
  原因: 当前工程中的 `metrics.py` 使用 `np.float`
- `setuptools==75.8.0`
  原因: `pytorch-lightning 2.1.0` 导入路径需要 `pkg_resources`
- `wandb==0.13.5`
  原因: 当前训练入口默认使用 `WandbLogger`

## 安装过程中观察到的问题

1. 初次安装 `torch 2.5.1 + cu121` 时，磁盘空间被 `pip` 临时目录与缓存耗尽。
2. 清理 `/tmp/pip-build-env-*`、`/tmp/pip-unpack-*` 与 `/root/.cache/pip` 后，安装恢复正常。
3. 宿主机环境变量存在 `OMP_NUM_THREADS=0`，需要在运行时显式覆盖为 `1`。

## 验证结果

### 官方 Mamba CUDA 前向

- 结果: 通过
- 输入形状: `[2, 32, 64]`
- 输出形状: `[2, 32, 64]`
- 参数量: `32640`

### 当前工程冒烟测试

- 脚本: `scripts/smoke_test_cpu.py`
- 结果: 通过
- 覆盖内容:
  - 临时样本构造
  - `Circuitnet` 数据集读取
  - `collate_fn`
  - `CircuitFormer` 初始化

### 验证脚本闭环

- 脚本: `true_mamba_experiments/scripts/verify_true_mamba_env.sh`
- 结果: 通过
- 说明: 单条命令已经覆盖官方 `Mamba` CUDA 前向与仓库自带 CPU 冒烟测试

## 当前边界

- 当前验证只说明独立环境已经具备承载真正 Mamba 实验的条件
- 当前验证尚未覆盖真正 Mamba neck 接入后的整模前向
- 当前验证尚未覆盖 1-step 训练与完整单次 run
