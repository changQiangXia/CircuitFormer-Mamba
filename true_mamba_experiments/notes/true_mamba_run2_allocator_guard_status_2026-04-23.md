# True Mamba Run2 Allocator Guard Status 2026-04-23

## 当前运行目录

- `exp/congestion_true_mamba_run2_allocator_guard_2026-04-23_16-08-00_UTC`

## 当前进程

- `train.pid=25933`
- `monitor.pid=25934`
- 巡检间隔: `600` 秒

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

## run1 到 run2 的唯一新增项

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 首次人工巡检

- sanity check: 已通过
- 当前阶段: `Epoch 0`
- 早期 `loss_step`: 约 `23` 到 `33`
- GPU 进程显存: 约 `19.8 GB`
- 当前入口未出现新的导入或启动异常

## 当前判断

- run2 已顺利进入正式训练
- 当前数值路径保持有限
- allocator guard 路线已通过入口级验证
- 后续关键观察点为 run1 原 OOM 区间附近是否可平稳越过
