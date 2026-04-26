# True Mamba Scheme B Preflight 2026-04-24

## 关联清单

- `../../../cklists/circuitformer_true_mamba_scheme_b_2026-04-24_01-56-34_UTC.md`
- `../../../cklists/circuitformer_true_mamba_master_rebuild_2026-04-24_01-50-20_UTC.md`

## 运行目录

- `exp/preflight_true_mamba_scheme_b_2026-04-24_01-59-52_UTC`

## 当前口径

- 任务: congestion prediction
- 损失: weighted MSE
- 学习率: `0.001`
- seed: `3407`
- batch size: `8`
- precision: `bf16-mixed`
- gradient clip: `1.0`
- true Mamba 关键配置:
  - `downsample=4`
  - `use_input_norm=False`
  - `use_mask=True`
  - `mask_pool_mode=max`
  - `out_proj_init_zero=False`
  - `out_proj_init_std=0.001`
  - `use_residual_scale=True`
  - `residual_scale_init=0.001`
  - `remask_after_upsample=True`

## preflight 结果

- 训练入口通过
- 单 batch train loss 保持有限
- 单 batch validation 指标保持有限
- `Epoch 0` train `loss_step=20.40`
- `Epoch 0` validation:
  - `pearson=-0.05857910246207281`
  - `spearman=-0.0692438758401569`
  - `kendall=-0.04946190196518486`
- 已生成:
  - `last.ckpt`
  - `epoch=0-pearson=-0.0586.ckpt`

## 空白栅格语义诊断

- 单样本 encoder 输出:
  - `value_zero_frac=0.8084545135498047`
  - `empty_cell_frac=0.6021575927734375`
- 经过方案 B 的 mask 输入约束后:
  - `masked_input_zero_frac=0.8084545135498047`
- 当前观察:
  - 空白栅格零语义已保留
  - 当前路径未再出现“入口归一化把空白格整体改写为统一负值”的现象

## 下采样 mask 诊断

- pooled mask 形状: `(1, 1, 64, 64)`
- pooled mask active frac: `0.45166015625`
- 当前观察:
  - 当前 `downsample=4` 路径已经把 scan 作用域收束到更小的有效区域集合

## 首步梯度诊断

- `residual_scale_grad=4.648924800676468e-08`
- `outer_out_proj_grad_norm=4.9107111266266656e-08`
- `row_in_proj_grad_norm=4.704311384884363e-10`
- `row_out_proj_grad_norm=7.17984172116104e-10`
- `col_in_proj_grad_norm=5.046644213635432e-10`
- `col_out_proj_grad_norm=6.784139361393215e-10`
- 当前观察:
  - 内部 Mamba 参数首步梯度已从精确 `0` 变为非零
  - 当前梯度量级仍偏小，后续正式 run 需要继续观察早期唤醒速度

## checkpoint 有限性扫描

- `last.ckpt` 非有限张量数: `0`
- 当前观察:
  - 当前 preflight 阶段未出现 decoder BN running stats 污染

## 当前判断

- 方案 B 已完成“可导入 + 可训练 + 可验证 + checkpoint 保持有限”的入口验证
- 方案 B 已修复 run3 中最直接的入口零语义破坏问题
- 方案 B 已缓解 exact zero-init 带来的“内部梯度精确为零”问题
- 下一步适合进入正式前 `20` 个 epoch run
