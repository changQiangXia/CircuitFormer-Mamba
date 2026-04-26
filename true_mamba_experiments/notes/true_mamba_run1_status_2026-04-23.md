# True Mamba Run1 Status 2026-04-23

## 当前运行目录

- `exp/congestion_true_mamba_run1_2026-04-23_11-15-00_UTC`

## 当前进程

- `train.pid=13429`
- `monitor.pid=13430`
- 巡检间隔: `600` 秒
- `train.pid` 当前已结束

## 配置口径

- 任务: congestion prediction
- 主线损失: weighted MSE
- 学习率: `0.001`
- seed: `3407`
- batch size: `8`
- num_workers: `8`
- precision: `16-mixed`
- 真正 Mamba 配置:
  - `num_blocks=1`
  - `d_state=16`
  - `d_conv=4`
  - `expand=2`
  - `downsample=1`
  - `bidirectional=True`
  - `out_proj_init_zero=True`

## 启动记录

- 首次后台启动目录: `exp/congestion_true_mamba_run1_2026-04-23_11-12-56_UTC`
- 首次后台启动收束原因: `conda run` 默认输出捕获会削弱实时日志可见性
- 修正后正式启动目录: `exp/congestion_true_mamba_run1_2026-04-23_11-15-00_UTC`

## 首次人工巡检

- sanity check: 已通过
- 当前阶段: `Epoch 0`
- 早期 loss: 约 `20` 到 `36`
- GPU 进程显存: 约 `20.5 GB`
- 早期稳定步速: 约 `0.9 step/s`

## 当前判断

- run1 前半段训练日志稳定，数值路径保持有限
- run1 当前终止，终止原因已定位为 `torch.OutOfMemoryError`
- 最后可见进度点约为 `Epoch 7`, `490/1037`, 约 `47%`
- 触发日志记录为额外申请约 `1.92 GiB`
- 触发时 GPU 总显存约 `23.56 GiB`，空闲约 `1.52 GiB`
- 触发时 PyTorch 已分配约 `13.24 GiB`，已保留未分配约 `8.46 GiB`
- 当前目录尚无 `last.ckpt` 与 `epoch=*.ckpt`
- 当前结论指向单卡显存碎片化或余量偏紧风险已经进入主矛盾
- 下一步优先路线为保持配方冻结，仅加入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后 fresh rerun
