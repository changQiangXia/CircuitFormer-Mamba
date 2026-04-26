# Config Notes

本目录保存 true Mamba 路线的字段口径与训练模板。

## 当前文件

- `true_mamba_neck_template.yaml`
  记录 neck 接入阶段的字段范围
- `true_mamba_neck_run1_freeze.yaml`
  记录早期冻结式尝试的口径

## 当前对照原则

- 数据集划分保持与强基线一致
- 学习率保持 `0.001`
- 随机种子保持 `3407`
- 训练目标保持 weighted MSE 主线
- neck 插入位置保持 encoder 与 decoder 之间
