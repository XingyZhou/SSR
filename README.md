# SSR-Mamba：小波采样 + 光谱 SSM 的光谱快照重建

本仓库是在 CVPR 2024 论文 **SSR**（[Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Improving_Spectral_Snapshot_Reconstruction_with_Spectral-Spatial_Rectification_CVPR_2024_paper.html)）代码基础上的研究分支。
任务：CASSI 快照压缩成像的 28 波段高光谱重建（CAVE 训练、10 场景测试），3-stage 展开网络。
研究主题：把 Transformer 块换成 Mamba / 状态空间模型（SSM），并研究 **小波采样** 与 **沿光谱维扫描** 两种用法。

原版 SSR 的说明、权重和引用见文末「上游 SSR」。

## 当前状态（2026-10-28）

| 线 | 状态 |
|---|---|
| HyMCNetL3（小波上下采样 + 空间扫描 Mamba） | 训练完成，39.12 dB，是目前最好的 Mamba 版本，对照基准 |
| 块内分频 V3 / V4 / V5 / V6 | 训练完成，全部低于不分频基线，**已放弃** |
| **V7**（HyMCNetL3 骨架，Mamba 改为沿光谱维扫描） | **训练中**，epoch 229 / 300，训练 PSNR 42.97 dB，预计 2026-10-29 结束；尚未测试 |
| V7S（V7 + 固定光谱先验的 FixedSigmaMamba，三档消融） | 实现并验证完毕，未训练 |
| 128 波段 Chikusei 线（`model_nb` / `Train_nb`） | 数据、代码、冒烟全部就绪，未训练（等 GPU 空出） |

详细结果、消融链和下一步计划：[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
在新服务器上复现环境、取回数据和权重：[docs/SETUP.md](docs/SETUP.md)。

## 仓库结构

代码都在根目录（脚本之间用相对导入和相对路径，不要移动）。数据、检查点、日志、重建结果不入库（见 `.gitignore`），用 `scripts/pack_transfer.sh` 在服务器之间复制。

| 文件 | 作用 |
|---|---|
| `Model.py`、`Dataset.py`、`Utils.py`、`Train.py`、`Test.py`、`Test_pretrain.py` | 上游 SSR 原始代码。`Train.py` 现在默认训练 `Mamba_Model.HyMCNetL3`，检查点写到 `Checkpoint_dwt/` |
| `Mamba_Model.py` | Mamba 版本的公共模块（MamConvB、Haar 上下采样、ConvB、FFN）和 `HyMCNetL3`、`HyMCNetL3Parallel`、`HyMCNetL3_V2`、`HyMCNetL3_V3` |
| `model_baseline.py` | `HyMCNetL3_Base`：SSR 的步长卷积采样 + MamConvB 块（隔离采样层的作用） |
| `model_v4.py`、`model_v5.py`、`model_v6.py` | 块内小波双支路系列（已放弃，保留作消融证据） |
| `model_v7.py` | **V7**：`SpecMamba_Att`，8×8 窗口内沿通道（波长）扫描，其余与 HyMCNetL3 完全相同 |
| `model_v7s.py` | **V7S**：V7 的 Mamba 换成 `FixedSigmaMamba`（A、dt 固定为光谱先验，B/C 三档 `--bc_mode`） |
| `model_nb.py`、`Dataset_nb.py`、`Train_nb.py`、`Test_nb.py` | 任意波段数版本（HyMCNetL3_NB / V7NB / V7SNB），用于 128 波段 Chikusei |
| `prepare_chikusei.py` | 把 Chikusei 原始立方体切成本仓库的场景格式，并生成 128 波段掩膜 |
| `Train_v{2..7}.py`、`Train_v7s.py`、`Train_base.py`、`Train_parallel.py` | 各版本训练脚本，只在模型类和检查点目录上不同；自动从最新检查点续训 |
| `Test_v3.py` | 通用测试脚本（名字是历史原因），`--model <类名> --ckpt_dir --out_dir --epochs 291-300`，输出 `summary.csv / per_scene.csv / report.md / best.json` |
| `analyze_freq.py`、`swap_bands.py` | 诊断工具：频段误差分解、两模型子带互换（oracle） |
| `Cal_para_flops.py`、`make_figure.py`、`find_best_blue_pixel.py`、`4_3stage_test.py` | 参数量 / FLOPs、论文图、早期测试脚本 |
| `monitor_train.py`、`run_v7.sh`、`run_chik.sh`、`run_queue.sh` | tmux 训练队列 + 微信（Server酱）/ Bark 进度推送 |
| `SpectralMambaNative/` | `FixedSigmaMamba` 模块、单元测试、以及接在原版 SSR `Net` 上的旧消融线（[说明](SpectralMambaNative/README.md)） |
| `docs/` | 实验总览、新服务器安装、100 波段线说明、每周进展报告 |
| `周报ppt/` | 日文进捗报告 pptx 生成脚本（模板和生成的 pptx 不入库） |

## 快速开始

环境安装见 [docs/SETUP.md](docs/SETUP.md)。以下命令都在仓库根目录、`.venv` 激活后执行。

```bash
# 训练（自动续训；检查点目录在脚本里固定：Train_v7.py -> ./Checkpoint_v7）
python Train_v7.py

# 带微信推送和日志的 tmux 方式
export SCT_KEY=SCTxxxx            # Server酱 SendKey，可选
tmux new -s train_v7 -d 'bash run_v7.sh'

# 测试：评估 291-300 轮，选测试 PSNR 最高的一轮
python Test_v3.py --model HyMCNetL3_V7 --ckpt_dir ./Checkpoint_v7 --out_dir ./Result_v7 --epochs 291-300

# 频段诊断 / 子带互换（模型写法 名字:模块.类:权重）
python analyze_freq.py --models SSR-S:Model.Net:./Checkpoint_pretrain/model_S.pkl V7:model_v7.HyMCNetL3_V7:./Checkpoint_v7/model_300.pkl
python swap_bands.py --a V7:model_v7.HyMCNetL3_V7:./Checkpoint_v7/model_300.pkl --b HyMCNetL3:Mamba_Model.HyMCNetL3:./Checkpoint_dwt/model_300.pkl --levels 1 2 3 4

# V7S 三档（σ 以波段计）
python Train_v7s.py --bc_mode selective --sigma 4
python Train_v7s.py --bc_mode s4 --sigma 4
python Train_v7s.py --bc_mode const --sigma 4

# 128 波段线（见 docs/100bands.md）
tmux new -s train_chik -d 'bash run_chik.sh'
```

## 主结果（CAVE 10 场景测试集，300 epoch）

| 模型 | 结构要点 | 参数量 | PSNR (dB) | SSIM | SAM (°) |
|---|---|---|---|---|---|
| SSR-S（原版，3 stage） | SSRB 块 + 步长卷积采样 | 1.81M | 38.93 | 0.9731 | 4.20 |
| SSR-M（原版，6 stage） | 同上 | 3.62M | 39.94 | 0.9777 | 3.67 |
| **HyMCNetL3** | MamConvB 块 + **Haar 小波上下采样** | 2.94M | **39.12** | 0.9742 | 3.78 |
| Base | MamConvB 块 + SSR 步长卷积采样 | 2.04M | 38.75 | 0.9724 | 4.00 |
| V6 | 块内小波双支路 + Hilbert 扫描 + 结构 token | 3.15M | 38.64 | 0.9712 | 4.21 |
| V5 | 块内小波双支路 + SSR 初始化 | 2.90M | 38.51 | 0.9706 | 4.15 |
| V4 | 块内小波双支路 | 2.04M | 38.21 | 0.9690 | 4.40 |
| V3 | 块内分解、无满分辨率块 | 1.25M | 36.55 | 0.9570 | 5.64 |
| V7 | HyMCNetL3 + 光谱维扫描 | 2.95M | 训练中 | | |

三条结论：小波用于上下采样层 +0.37 dB 且不增参数；块内分频一律负收益；Mamba 的增益全部落在低频（64×64 以下）。
证据与分析见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 和 [docs/weekly/](docs/weekly/)。

## 文档

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：所有版本的谱系、检查点、结果、消融链、诊断结论、待办
- [docs/SETUP.md](docs/SETUP.md)：新服务器环境、数据下载、权重迁移、训练监控
- [docs/TRAIN_V7S.md](docs/TRAIN_V7S.md)：V7S 三档消融的训练手册（检查、冒烟、启动、测试、对比、结果解读）
- [docs/100bands.md](docs/100bands.md)：128 波段 Chikusei 线的数据处理、显存、启动方式
- [docs/weekly/](docs/weekly/)：每周进展报告（2026-10-15、2026-10-19）
- [SpectralMambaNative/README.md](SpectralMambaNative/README.md)：FixedSigmaMamba 的理论、实现与测试

---

## 上游 SSR

以下为原仓库 [SSR](https://github.com/XingyZhou/SSR) 的说明，原始代码、预训练权重（`Checkpoint_pretrain/model_{S,M,L}.pkl`）和测试流程保持不变。

### 1. Environment Requirements
```shell
Python>=3.6
scipy
numpy
einops
```

### 2. Train

Download the cave dataset of [MST series](https://github.com/caiyuanhao1998/MST) from [Baidu disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ)`code:fo0q` or [here](https://pan.baidu.com/s/1gyIOfmUWKrjntKobUjwTjw?pwd=lup6), put the dataset into the corresponding folder "SSR/CAVE_1024_28/" as the following form:

	|--CAVE_1024_28
        |--scene1.mat
        |--scene2.mat
        ：
        |--scene205.mat
        |--train_list.txt
Then run the following command
```shell
cd SSR
python Train.py
```

### 3. Test

Download the test dataset from [here](https://pan.baidu.com/s/1KqMo3CY8LU9HRU2Lak9yfQ?pwd=c0a2), put the dataset into the corresponding folder "SSR/Test_data/" as the following form:

	|--Test_data
        |--scene01.mat
        |--scene02.mat
        ：
        |--scene10.mat
        |--test_list.txt
Then run the following command
```shell
cd SSR
python Test.py
```
For testing pre-trained models, run the following command
```
python Test_pretrain.py
```
Finally, run 'cal_psnr_ssim.m' in Matlab to get the performance metrics.

### Citation
If this repo helps you, please consider citing our work:

```shell
@InProceedings{SSR,
    author    = {Zhang, Jiancheng and Zeng, Haijin and Chen, Yongyong and Yu, Dengxiu and Zhao, Yin-Ping},
    title     = {Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {25817-25826}
}
```
