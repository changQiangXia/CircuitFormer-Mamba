# True Mamba NaN Incident 2026-04-23

## 关联运行目录

- `exp/congestion_true_mamba_run2_allocator_guard_2026-04-23_16-08-00_UTC`

## 当前现象

- run2 已跨过此前 `Epoch 7` 中段的 OOM 区间
- `Epoch 9` 结束后已完成首轮正式验证
- 首轮正式验证打印结果为 `pearson: nan spearman: nan kendall: nan`
- 当前目录已生成 `last.ckpt`
- 当前目录已生成 `epoch=9-pearson=nan.ckpt`
- 训练随后继续进入 `Epoch 10`

## 已知补充信号

- `launcher.log` 中可见 `Epoch 5` 起已有若干 `loss_step=nan.0`
- `launcher.log` 中可见 `Epoch 5` 后 `loss_epoch` 长期显示为 `nan.0`
- 因此当前异常范围不局限于验证指标展示

## 当前判断

- run2 当前更适合视作“显存路线打通 + 数值异常暴露”的排障 run
- run2 当前不宜直接写入正式对比结论
- 下一步应先停止当前 run，随后利用 `last.ckpt` 做离线验证诊断
