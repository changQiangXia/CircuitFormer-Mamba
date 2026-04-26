# CircuitFormer-Mamba

`CircuitFormer-Mamba` 记录了 `CircuitFormer` 的高质量复现、工程修复，以及两条 Mamba 相关 neck 路线的并行探索与对照。公开结果围绕 `Circuit as Set of Points` 的 congestion prediction 路线整理，原论文见 <https://arxiv.org/abs/2310.17418>。

## 当前定位

- 高质量复现: 原版 `CircuitFormer` 训练流程已跑通。
- 工程修复: 数据路径、训练配方、中断恢复、测试归档与分析脚本已整理到可复核状态。
- 并行探索与对照:
  - `zero-init BEV Mamba`: 位于 `model/bev_mamba.py` 的轻量二维残差 neck，参数增量约 `+0.125%`。
  - `true Mamba Scheme B`: 位于 `true_mamba_experiments/modules/true_mamba_neck.py` 的官方 `mamba_ssm.Mamba` neck 路线。

## 当前结果快照

| 方案 | Val Pearson | Test Pearson | 说明 |
| --- | ---: | ---: | --- |
| 原版首个完整复现 | 0.5570 | 0.5409 | 原始训练配方跑通 |
| 强基线重跑 | 0.6488 | 0.6382 | 训练配方修复后的对照基线 |
| zero-init BEV Mamba | 0.6499 | 0.6404 | 三项测试指标均高于强基线 |
| true Mamba Scheme B | 0.6467 | 0.6358 | Pearson 接近强基线，Spearman / Kendall 高于强基线 |

## 环境矩阵

| 用途 | conda 环境 | Python | Torch 栈 | 依赖清单 |
| --- | --- | --- | --- | --- |
| 主线复现与 `zero-init BEV Mamba` | `circuitformer-gpu` | `3.9.25` | `torch 1.13.0+cu117` / `torchvision 0.14.0+cu117` / `torchaudio 0.13.0+cu117` | `requirements.mainline.txt` |
| `true Mamba Scheme B` | `circuitformer-true-mamba` | `3.10.20` | `torch 2.5.1+cu121` / `torchvision 0.20.1+cu121` / `torchaudio 2.5.1+cu121` | `true_mamba_experiments/requirements-true-mamba.txt` |

`requirements.txt` 作为历史快照保留。当前服务器上完成验证的依赖集合以上表两份清单为准。

## 主线复现

数据集下载说明: <https://circuitnet.github.io/intro/download.html>

默认数据配置位于 `config/config.yaml`:

- `data.data_root=${hydra:runtime.cwd}/../datasets/CircuitNet-N28/graph_features/instance_placement_micron`
- `data.label_root=${hydra:runtime.cwd}/../datasets/CircuitNet-N28/training_set/congestion/label`

主线复现命令如下:

```bash
conda create -n circuitformer-gpu python=3.9 -y
conda activate circuitformer-gpu
source /etc/network_turbo || true
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.mainline.txt
pip install spconv-cu117==2.3.6
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
mkdir -p ckpts
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O ckpts/resnet18.pth
```

训练与测试入口:

```bash
WANDB_MODE=disabled python train.py experiment.save_dir=exp/<run_name>

WANDB_MODE=disabled python train.py \
  model.bev_mamba.enabled=true \
  model.bev_mamba.out_proj_init_zero=true \
  experiment.save_dir=exp/<run_name>

WANDB_MODE=disabled python test.py \
  experiment.ckpt_path=exp/<run_name>/<model>.ckpt
```

## True Mamba 独立环境记录

独立环境的设立基于两点事实:

- 主线复现已经在 `torch 1.13.0+cu117` 上稳定归档。
- 官方 `mamba-ssm` 在本机完成验证的组合为 `torch 2.5.1+cu121 + causal-conv1d 1.6.1 + mamba-ssm 2.3.1`。

环境与训练阶段遇到的关键问题及处理如下:

| 环节 | 现象 | 处理 |
| --- | --- | --- |
| 大体积 wheel 安装 | `torch 2.5.1+cu121` 与编译依赖安装时，`pip` 临时目录和缓存目录带来额外磁盘压力 | 使用 `source /etc/network_turbo || true`、`--no-cache-dir`，磁盘紧张时清理 `/tmp/pip-build-env-*`、`/tmp/pip-unpack-*` 与 `/root/.cache/pip` |
| 宿主机线程环境变量 | 本机曾出现 `OMP_NUM_THREADS=0`，`libgomp` 会直接拒绝启动 | 训练、验证、安装脚本统一显式设置 `OMP_NUM_THREADS=1` |
| `numpy` 版本 | `metrics.py` 仍使用 `np.float` | true Mamba 环境固定 `numpy==1.23.5` |
| `setuptools` 版本 | `pytorch-lightning 2.1.0` 的导入路径仍会访问 `pkg_resources` | true Mamba 环境固定 `setuptools==75.8.0` |
| 全分辨率 true Mamba 扫描 | `downsample=1` 路线在显存与早期训练表现上都偏紧 | Scheme B 改用 `downsample=4`，并在有效区域上做 mask 约束 |
| `fp16-mixed` 数值稳定性 | 早期 true Mamba run 出现 `loss_step=nan.0`，坏 checkpoint 中出现非有限 `BatchNorm` 统计量 | 后续入口切换为 `bf16-mixed`，加入 finite guard 与 `gradient_clip_val=1.0` |
| exact zero-init 梯度唤醒 | `out_proj` 精确置零时，内部 Mamba 参数首步梯度过小 | Scheme B 采用 `out_proj_init_std=0.001` 与 `residual_scale_init=0.001` |

当前服务器上已验证的 true Mamba 环境创建脚本与验证脚本如下:

```bash
bash true_mamba_experiments/scripts/create_true_mamba_env.sh
bash true_mamba_experiments/scripts/verify_true_mamba_env.sh
```

训练入口:

```bash
bash true_mamba_experiments/scripts/train_congestion_true_mamba_scheme_b.sh
```

该脚本对应的稳定化设置包括:

- `trainer.precision=bf16-mixed`
- `trainer.gradient_clip_val=1.0`
- `model.true_mamba.downsample=4`
- `model.true_mamba.use_mask=True`
- `model.true_mamba.out_proj_init_zero=False`
- `model.true_mamba.out_proj_init_std=0.001`
- `model.true_mamba.use_residual_scale=True`
- `model.true_mamba.residual_scale_init=0.001`

## 范围说明

- 当前公开结论对应 congestion prediction 路线。
- 当前训练主线为带像素权重的 MSE。
- `config/config.yaml` 中保留 DRC 入口，当前公开结果未扩展到独立 DRC 实验线。
- `model.bev_mamba.enabled` 与 `model.true_mamba.enabled` 在 `model/circuitformer.py` 中互斥。
- Pearson / Spearman / Kendall 采用“逐样本计算，再对样本平均”的口径。
- 当前仓库路线包含 `resnet18` 预训练先验，权重文件路径为 `ckpts/resnet18.pth`。
- 当前对照证据由单次强基线、单次 `zero-init BEV Mamba`、单次 `true Mamba Scheme B` 共同构成。

## 建议阅读顺序

1. `analysis/acceptance_2026-04-15/project_closeout_report.md`
2. `true_mamba_experiments/README.md`
3. `true_mamba_experiments/ENVIRONMENT.md`
4. `true_mamba_experiments/notes/`
