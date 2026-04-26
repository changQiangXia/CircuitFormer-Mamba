# True Mamba Prototype Smoke 2026-04-23

## 目标

在保持主线训练入口原样的前提下，先验证一个隔离原型:

- 当前真实 neck 入口形状是否已经明确
- 官方 `Mamba` 组成的 neck 级原型能否完成模块级前向与反向

## 形状审计结果

- 脚本: `true_mamba_experiments/scripts/smoke_shape_audit.py`
- 设备: `cuda`
- 当前 baseline neck: `Identity`
- `Encoder` 输出形状: `[2, 64, 256, 256]`
- neck 输出形状: `[2, 64, 256, 256]`
- `decoder` 输出形状: `[2, 1, 256, 256]`
- 当前整模总参数量: `23640401`

## 原型模块说明

- 文件: `true_mamba_experiments/modules/true_mamba_neck.py`
- 名称: `TrueMambaNeckPrototype`
- 结构:
  - `GroupNorm(1, 64)`
  - 行方向官方 `Mamba`
  - 列方向官方 `Mamba`
  - 双向序列混合
  - `1x1` 输出投影
  - 残差回加

## 模块级 smoke 结果

- 脚本: `true_mamba_experiments/scripts/smoke_true_mamba_neck_module.py`
- 输入形状: `[2, 64, 256, 256]`
- 输出形状: `[2, 64, 256, 256]`
- 参数量: `69504`
- 前向耗时: `0.229018 s`
- 反向耗时: `0.266476 s`
- 峰值显存: `1826.64 MB`

## 当前结论

- 当前真正 Mamba neck 原型已完成模块级 smoke
- 当前真正 Mamba neck 已通过可选 builder 接入 `CircuitFormer`
- 当前默认 baseline 路径与当前轻量 `BEVMambaNeck` 路径均已完成回归检查
- 下一步重点转向首轮 run 配置冻结与正式实验启动

## 接线后的整模 smoke 结果

- 脚本: `true_mamba_experiments/scripts/smoke_true_mamba_full_model.py`
- neck 模块: `TrueMambaNeck`
- 输出形状: `[2, 1, 256, 256]`
- 整模总参数量: `23709905`
- neck 参数量: `69504`
- 前向耗时: `0.512554 s`
- 峰值显存: `717.61 MB`

## 接线后的 1-step 训练结果

- 脚本: `true_mamba_experiments/scripts/smoke_true_mamba_one_step.py`
- neck 模块: `TrueMambaNeck`
- loss: `633.401306`
- 训练前向阶段耗时: `0.522855 s`
- 反向阶段耗时: `0.406944 s`
- 峰值显存: `2573.97 MB`

## AMP 稳定性补充

- 首次 `train.py` 级 preflight 显示，真正 Mamba 在 `16-mixed` 路径上的首个真实 batch 会出现 `loss=nan`
- 真实 batch 定位结果显示:
  - `baseline + fp32`: 有限
  - `baseline + amp16`: 有限
  - `true_mamba + fp32`: 有限
  - `true_mamba + amp16`: 初始版本出现全张量 `nan`
- 当前处理方式为在 `true_mamba_experiments/modules/true_mamba_neck.py` 中，将 Mamba 行列扫描局部固定在 `fp32`，输出再回写到外层 dtype
- 修复后，`true_mamba + amp16` 的真实 batch 输出与 loss 均恢复有限
- 修复后，`preflight_true_mamba_train_entry.sh` 已完整通过，`loss_step=26.60`

## 分支回归结果

- baseline 默认路径: 通过
- 轻量 `BEVMambaNeck` 路径: 通过
- 真正 `TrueMambaNeck` 路径: 通过模块级 smoke、整模前向、1-step 训练
