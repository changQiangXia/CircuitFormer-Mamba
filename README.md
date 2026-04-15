# CircuitFormer

本仓库包含两部分内容:

- 原始 `CircuitFormer` 训练与测试代码
- 本项目收尾阶段使用的 `BEV Mamba` 验收材料

公共仓库默认不包含数据集、训练输出目录、实验 checkpoint 与下载得到的预训练权重。

原论文: *Circuit as Set of Points*，NeurIPS 2023，论文链接为 <https://arxiv.org/abs/2310.17418>。

## 运行前提

- 命令默认从仓库根目录执行。
- 建议环境为 `Python 3.9`。
- 训练与正式测试默认面向 `CUDA` 单卡环境。
- `torch`、`torchvision`、`torchaudio` 需按本机 CUDA 环境安装。
- 额外依赖包括 `torch_scatter`、`spconv` 与 `requirements.txt` 中列出的包。
- `ckpts/resnet18.pth` 必须存在；`model/circuitformer.py` 会在构建解码器时加载该文件。
- 若不使用 Weights & Biases，建议设置 `WANDB_MODE=disabled`；当前代码在该模式下会回退到本地 `CSVLogger`。

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

`scripts/*.sh` 默认调用名为 `circuitformer` 的 conda 环境；若环境名不同，可通过 `CONDA_ENV_NAME` 覆盖。

## 建议阅读顺序

1. `README.md`: 仓库定位、依赖、数据路径与入口命令
2. `analysis/acceptance_2026-04-15/project_closeout_report.md`: 复现过程、结果图表与结项结论
3. `scripts/`: 训练、恢复与分析辅助脚本
