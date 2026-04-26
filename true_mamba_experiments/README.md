# True Mamba Experiments

本目录记录 `true Mamba Scheme B` 路线的独立环境、实现、训练脚本与阶段性记录。目录目标较明确: 在保持 `CircuitFormer` 主干与强基线训练配方尽量稳定的前提下，将官方 `mamba_ssm.Mamba` 接到 encoder 与 decoder 之间的 neck 位置，并与强基线、轻量 `zero-init BEV Mamba` 路线做同口径对照。

## 当前结论

- 独立环境 `circuitformer-true-mamba` 已完成安装与验证。
- 官方 `mamba-ssm` 与 `causal-conv1d` 已在本机完成 CUDA 前向验证。
- `true_mamba_experiments/modules/true_mamba_neck.py` 已完成接线。
- `true Mamba Scheme B` 单次正式 run 已完成，归档目录为 `exp/congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC`。
- 当前单次 run 的最终指标形态为: Pearson 接近强基线，Spearman / Kendall 高于强基线。

## 环境与任务边界

本目录中的内容默认建立在 `circuitformer-true-mamba` 之上，职责边界与主线环境不同。

- `circuitformer-gpu`
  - 负责原版 `CircuitFormer`、强基线、`zero-init BEV Mamba`
  - 更适合主线复现与轻量 neck 对照
- `circuitformer-true-mamba`
  - 负责 `true Mamba Scheme B`
  - 更适合 official-Mamba 依赖安装、preflight、正式训练与诊断

若主线环境尚未跑通，更稳妥的顺序是先完成主线复现，再切换到本目录对应的独立环境。

## 对照对象

- 强基线: `exp/congestion_rerun_lr1e-3_seed3407_2026-04-12_10-58-54_UTC`
- 轻量 neck: `exp/congestion_bev_mamba_zero_init_2026-04-13_19-08-30_UTC`
- official-Mamba neck: `exp/congestion_true_mamba_scheme_b_2026-04-24_02-03-01_UTC`

## 目录说明

- `modules/`: official-Mamba neck 实现
- `configs/`: 试验模板与字段口径
- `scripts/`: 环境创建、验证、preflight、训练、恢复与诊断脚本
- `notes/`: 环境记录、数值稳定性排障记录、preflight 与 run 状态记录

## 推荐阅读顺序

1. `ENVIRONMENT.md`
2. `notes/environment_validation_2026-04-23.md`
3. `notes/true_mamba_scheme_b_preflight_2026-04-24.md`
4. `modules/true_mamba_neck.py`
5. `scripts/train_congestion_true_mamba_scheme_b.sh`
