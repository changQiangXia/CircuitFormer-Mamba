# CircuitFormer-Mamba

本仓库包含两部分内容:

- 原始 `CircuitFormer` 训练与测试代码
- 本项目收尾阶段使用的 `BEV Mamba` 验收材料

公共仓库默认剔除数据集、训练输出目录、实验 checkpoint 与下载得到的预训练权重。

仓库中使用的 `BEV Mamba` 属于项目简称。当前实现更准确的定位可表述为：在 BEV 特征图上插入一个**受 Mamba / selective state space 思想启发的轻量二维残差 neck**。模块作用范围集中在 encoder 与 decoder 之间的 neck 级增量改造。

原论文: *Circuit as Set of Points*，NeurIPS 2023，论文链接为 <https://arxiv.org/abs/2310.17418>。

## 项目定位

- 高质量复现: 原版 `CircuitFormer` 训练流程已打通。
- 工程修复: 训练脚本、路径配置、中断恢复与验收材料已整理到可复核状态。
- 探索性改进: 在强基线之上加入一个 `Mamba-inspired` 的轻量 `BEV residual neck`，当前观察到小幅正增益，参数增量约 `+0.125%`。

## 运行前提

- 命令默认从仓库根目录执行。
- 建议环境为 `Python 3.9`。
- 训练与正式测试默认面向 `CUDA` 单卡环境。
- `torch`、`torchvision`、`torchaudio` 需按本机 CUDA 环境安装。
- 额外依赖包括 `torch_scatter`、`spconv` 与 `requirements.txt` 中列出的包。
- `ckpts/resnet18.pth` 必须存在；`model/circuitformer.py` 会在构建解码器时加载该文件，因此当前路线包含 `resnet18` 预训练先验。
- 未启用 Weights & Biases 时，建议设置 `WANDB_MODE=disabled`；当前代码在该模式下会回退到本地 `CSVLogger`。

可参考如下安装顺序:

```bash
conda create -n circuitformer python=3.9 -y
conda activate circuitformer
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir -p ckpts
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O ckpts/resnet18.pth
```

## 数据路径

默认数据配置位于 `config/config.yaml`:

- `data.data_root=../datasets/CircuitNet-N28/graph_features/instance_placement_micron`
- `data.label_root=../datasets/CircuitNet-N28/training_set/congestion/label`
- 数据划分文件为 `data/train.txt`、`data/val.txt`、`data/test.txt`

数据集下载说明: <https://circuitnet.github.io/intro/download.html>

若数据集位于其他目录，可在命令行覆盖 `data.data_root` 与 `data.label_root`。

## 训练与测试

基线训练:

```bash
WANDB_MODE=disabled python train.py
```

启用 `BEV Mamba`:

```bash
WANDB_MODE=disabled python train.py \
  model.bev_mamba.enabled=true \
  model.bev_mamba.out_proj_init_zero=true \
  experiment.save_dir=exp/<run_name>
```

从中断点恢复训练:

```bash
WANDB_MODE=disabled python train.py \
  experiment.resume_ckpt_path=exp/<run_name>/last.ckpt
```

测试指定 checkpoint:

```bash
WANDB_MODE=disabled python test.py experiment.ckpt_path=exp/<run_name>/<model>.ckpt
```

`scripts/*.sh` 默认调用名为 `circuitformer` 的 conda 环境；环境名变化时，可通过 `CONDA_ENV_NAME` 覆盖。

## 范围说明

- 当前公共验收结论对应 **congestion prediction** 路线。
- `config/config.yaml` 中保留了 DRC label 路径入口；当前公开结果的叙述范围限定在 congestion prediction。
- `model/model_interface.py` 的当前训练主线使用带像素权重的 MSE；`model.loss` 字段在配置中保留，README 的表述范围据此限定在现有训练主线。
- 报告中的 Pearson / Spearman / Kendall 采用“逐样本计算，再对样本平均”的口径；该口径区别于将全数据集像素摊平后计算单次全局相关系数。
- 当前 `BEV Mamba` 相对强基线的结论可表述为**观察到的小幅正增益**；当前证据由 congestion prediction、weighted MSE 主线、单次强基线与单次 `zero-init` run 共同构成。

## 建议阅读顺序

1. `README.md`: 仓库定位、依赖、数据路径与入口命令
2. `analysis/acceptance_2026-04-15/project_closeout_report.md`: 复现过程、结果图表与结项结论
3. `scripts/`: 训练、恢复与分析辅助脚本
