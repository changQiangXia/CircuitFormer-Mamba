# True Mamba Experiment Checklist

- [x] 明确“真正 Mamba”的实现口径
- [x] 确认 `mamba_ssm` 依赖可用
- [x] 选定最小改动路线
- [x] 完成官方模块级前向 smoke test
- [x] 完成项目环境级 smoke test
- [x] 保持强基线训练配方不变
- [x] 完成单次训练 run
- [x] 与强基线、轻量 neck 做同口径对比
- [x] 记录训练稳定性、显存、速度与指标变化
- [x] 完成继续推进与止损边界判断

## 当前已验证事实

- 独立环境名为 `circuitformer-true-mamba`
- 官方组合 `torch 2.5.1 + cu121 / causal-conv1d 1.6.1 / mamba-ssm 2.3.1` 可导入并可完成 CUDA 前向
- 当前工程在该环境下需要固定 `numpy 1.23.5`
- 当前工程在该环境下需要固定 `setuptools 75.8.0`
- 当前宿主机建议显式设置 `OMP_NUM_THREADS=1`
- `true Mamba Scheme B` 单次正式 run 已完成
- 测试集最终指标为:
  - Pearson `0.6358`
  - Spearman `0.4846`
  - Kendall `0.3656`

## 当前判断

- official-Mamba neck 路线已完成“可导入、可训练、可验证、可归档”的闭环
- 当前单次 run 呈现“Pearson 接近强基线，Spearman / Kendall 高于强基线”的指标形态
- 后续若继续推进，更合适的方向为多 seed、速度/显存统计与参数量统一统计
