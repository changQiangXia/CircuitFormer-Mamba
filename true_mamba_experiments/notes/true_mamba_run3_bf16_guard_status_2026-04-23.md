# True Mamba Run3 BF16 Guard Status 2026-04-23

## 关联清单

- `../../../cklists/circuitformer_true_mamba_run3_bf16_guard_2026-04-23_19-22-24_UTC.md`

## 背景

- run2 在 `fp16-mixed` 路径下于 `Epoch 5` 出现若干 `loss_step=nan.0`
- run2 的 `last.ckpt` 中共有 `84` 个非有限张量
- 这 `84` 个张量全部为 `BatchNorm` 的 `running_mean` 或 `running_var`
- 离线对照显示，坏 `checkpoint` 在 `eval()` 下 `encoder` 与 `true_mamba neck` 输出仍保持有限，`decoder` 输出已全量非有限
- 同一坏 `checkpoint` 在 `train()` 下，采样 batch 的整模输出保持有限
- 当前判断更接近“偶发非有限前向污染 BN 统计量”

## run3 最小修复口径

- 训练与验证入口新增 finite guard
- `TrueMambaBlock` 内部新增阶段级 finite guard
- 混合精度从 `16-mixed` 切换为 `bf16-mixed`
- 新增 `gradient_clip_val=1.0`
- 其余真正 Mamba 结构超参数继续保持 run2 口径

## run3 目录

- `exp/congestion_true_mamba_run3_bf16_guard_2026-04-23_19-25-33_UTC`

## 启动信息

- `train.pid=39353`
- `monitor.pid=39354`
- 巡检间隔: `300` 秒

## 启动前验证

- `bf16` 设备支持: `True`
- GPU: `NVIDIA GeForce RTX 3090`
- `fast_dev_run` 已通过
- `fast_dev_run` 的 train loss 为 `26.60`
- `fast_dev_run` 的 validation 指标保持有限:
  - `pearson=-0.13624973672589405`
  - `spearman=-0.20942136070951853`
  - `kendall=-0.1477220590751404`

## 当前状态

- `sanity check` 已通过
- `sanity check` 指标保持有限:
  - `pearson=-0.08655447953377501`
  - `spearman=-0.13085271351228586`
  - `kendall=-0.09334009978741957`
- 训练已进入 `Epoch 0`
- 前几十步 `loss_step` 保持有限，量级约在 `21.7` 到 `50.8`
- GPU 显存占用约 `20.4 GB`
- `Epoch 0` 已完整结束，`loss_epoch` 保持有限，量级约为 `14.40`
- 当前已进入 `Epoch 1`
- `Epoch 9` 正式验证保持有限:
  - `pearson=0.4801909493695525`
  - `spearman=0.44138656544365423`
  - `kendall=0.32790883949217975`
- `Epoch 19` 正式验证保持有限:
  - `pearson=0.5085616936889453`
  - `spearman=0.43952211574663624`
  - `kendall=0.32539648452621817`
- `Epoch 19` 相对强基线仍存在明显差距:
  - `pearson` 差 `-0.041861`
  - `spearman` 差 `-0.052977`
  - `kendall` 差 `-0.037752`
- 相对强基线的 gap 在 `Epoch 9 -> 19` 间未收敛
- 当前更适合作为“稳定性修复成功 + full-res true Mamba 早期效果观察”的 run
- 当前已决定停止 run3，并转入 `downsample=4` 的单变量 rerun

## 当前观察重点

- run4 是否继续保持数值稳定
- run4 的 `Epoch 9` 与 `Epoch 19` 是否优于 run3
