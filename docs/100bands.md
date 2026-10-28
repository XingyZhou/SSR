# 100+ 波段光谱 SSM 实验线(2026-10-19 建立)

动机:CAVE 只有 28 个波段,沿光谱维扫描的 Mamba(V7 / V7S)序列太短,看不出 SSM 相对注意力 / 卷积的优势。
本线把同一套网络和训练配方搬到 128 波段数据上,序列长度 ×4.6,其余全部不变。

## 1. 数据:Chikusei(128 波段)

| 项 | 值 |
|---|---|
| 来源 | 东京大学 Space Application Laboratory,Headwall Hyperspec-VNIR-C 航空高光谱,2014-07-29 |
| 原始立方体 | 2517 × 2335 × 128,363–1018 nm,GSD 2.5 m,`HSI_raw/chikusei/`(MATLAB v7.3,1.4 GB zip 已下载) |
| 许可 | CC BY 3.0;论文须致谢 SAL/U-Tokyo 并引用 Yokoya & Iwasaki 2016 技术报告(`HSI_raw/chikusei/Chikusei_MATLAB/Readme.txt`) |
| 归一化 | 全局单一尺度 = 有效像素全波段 99.99 百分位(5779.0),裁到 [0, 1];不做逐波段归一化,保留光谱形状。得到的分布(中位数≈0.14,p99≈0.6,max 1)与 CAVE 相近 |
| 划分 | 顶部 1024 行 = 测试区,其余 = 训练区(HSI 超分文献的常用划分) |
| 测试集 | 21 个 256×256 tile(256 网格,只保留无零填充角落的 tile),`Chikusei_128/test/scene_r*_c*.mat`,key `img` float32 |
| 训练集 | 35 个 256×256 tile,`Chikusei_128/train/tile_r*_c*.mat`,key `img_expand` uint16(×65535),与 CAVE 格式一致 |
| 掩膜 | `mask_256_128.mat` / `mask_128_128.mat`:`mask_256_28.mat` 的二维掩膜(28 层本来就相同)复制到 128 层;128 版是 256 版左上角 |
| 元数据 | `Chikusei_128/meta.json`(scale、tile 坐标) |

生成命令(已跑过,`Chikusei_128/prepare.log`):

```bash
.venv/bin/python prepare_chikusei.py --train_size 256
```

注意:512×512 的训练 tile 只能切出 6 块(航带是斜的,角落全零),所以改用 256 tile(35 块,约 230 万像素)。
训练 patch 是 128×128,随机裁剪 + 翻转旋转 + 四拼接增强仍然有效。

## 2. 代码(根目录,原文件一律未改)

| 文件 | 说明 |
|---|---|
| `model_nb.py` | 任意波段数的三个模型:`HyMCNetL3_NB`(空间扫描,对照)、`HyMCNetL3_V7NB`(光谱扫描,原生 S6)、`HyMCNetL3_V7SNB`(FixedSigmaMamba,`--bc_mode/--sigma/--dt_fixed`)。波段数、U-Net 宽度(`--mult`,默认 `1,2,4` 即 nC/2nC/4nC)、色散步长 `--len_shift`、窗口 `--win` 都由 opt 决定。块(MamConvB / MamConvB_V7 / MamConvB_V7S)、Haar 上下采样、ConvB、FFN 直接从 `Mamba_Model.py` / `model_v7.py` / `model_v7s.py` 导入 |
| `Dataset_nb.py` | `Dataset.py` 的泛化:场景可以是任意尺寸列表,uint16 常驻内存、按裁剪转 float;掩膜按 nC 复制、按 patch 裁剪 |
| `Train_nb.py` | `Train_v7.py` 的泛化,配方相同(Adam 4e-4,余弦到 1e-6,RMSE 多阶段损失 1.3/0.7/0.5,5000 样本/epoch);多了 `--model --data_path --ckpt_dir --mask_path --bands --mult --size --batch_size --sched_epochs` |
| `Test_nb.py` | `Test_v3.py` 的泛化,输出 `summary.csv / per_scene.csv / report.md / best.json / recon/` |
| `prepare_chikusei.py` | 见 §1,h5py 按行块读取(机器只有 31 GB 内存,V7 训练常驻 24 GB) |
| `run_chik.sh` | tmux 队列:先 V7NB 再 NB 对照,各带 `monitor_train.py` 微信推送 |

**正确性验证(已做)**

- 28 波段下 `HyMCNetL3_NB` / `HyMCNetL3_V7NB` 与 `HyMCNetL3` / `HyMCNetL3_V7` 参数量相同(2.9379 M / 2.9520 M),state_dict 键完全相同;
- 把 `Checkpoint_v7/model_038.pkl` 和 `Checkpoint_dwt/model_300.pkl` 分别载入新旧模型,同一输入输出逐位相等(三阶段 max|diff| = 0)。即 `model_nb` 是原模型的无损泛化,CAVE 结果可直接对照;
- 128 波段端到端:`Train_nb.py`(60 样本 1 epoch)→ 检查点 → `Test_nb.py` 21 景测试 → report,全流程通过(smoke 产物已删)。

## 3. 显存 / 速度(RTX 4090,V7 训练同时占着 11 GB,所以时间偏慢)

| 模型 | `--mult` | batch | patch | 参数 | 峰值显存 | s/step | 5000 样本/epoch |
|---|---|---|---|---|---|---|---|
| V7NB(光谱扫描) | 1,2,4 | 2 | 128 | 45.9 M | 11.7 GB | 1.16 | ≈48 min |
| V7NB | 1,2,4 | 1 | 128 | 45.9 M | 6.2 GB | 0.44 | ≈37 min |
| V7NB | 1,1,2 | 2 | 128 | 15.2 M | 8.8 GB | 1.02 | ≈43 min |
| V7NB | 1,1,1 | 2 | 128 | 9.8 M | 8.5 GB | 1.01 | ≈42 min |
| NB(空间扫描对照) | 1,2,4 | 2 | 128 | 53.8 M | 11.7 GB | 1.20 | ≈50 min |
| NB | 1,1,1 | 2 | 128 | 11.1 M | 8.4 GB | 1.02 | ≈43 min |
| (参考)V7,28 波段 | 1,2,4 | 2 | 256 | 2.95 M | 9.9 GB | — | ≈10 min |

- 128 波段下 256×256 patch 在 13 GB 空余显存里全部 OOM(即使 `--mult 1,1,1`),所以训练 patch 用 128;测试仍用 256×256 tile(网络是全卷积 + 8×8 窗口,尺寸无关)。
- 参数量随通道数平方增长(ConvB / FFN / 1×1 卷积),`1,2,4` 下 46 M 已经不是"轻量模型";`1,1,2`(15 M)是折中。宽度对速度几乎没影响,时间主要花在逐窗口的光谱扫描和下采样支路里的空间 Mamba。
- 300 epoch ≈ 10 天,不现实;`run_chik.sh` 默认 `EPOCHS=100`(约 3 天,余弦调度按 100 收尾)。GPU 空出来后(V7 预计 10-21 结束)速度会快一些。

## 4. 启动

```bash
# GPU 空出后(或接受 V7 变慢):
tmux new -s train_chik -d 'bash run_chik.sh'          # V7NB 100 epoch -> NB 100 epoch
EPOCHS=300 tmux new -s train_chik -d 'bash run_chik.sh'  # 与 CAVE 同一调度

# 单独跑一臂 / 省显存:
.venv/bin/python Train_nb.py --model HyMCNetL3_V7NB --data_path ./Chikusei_128/ --bands 128 --size 128 \
    --mask_path ./mask_128_128.mat --batch_size 1 --mult 1,1,2 --max_epochs 100 --sched_epochs 100 --ckpt_dir ./Checkpoint_chik_v7nb_m112

# 测试(21 景,256×256):
.venv/bin/python Test_nb.py --model HyMCNetL3_V7NB --data_path ./Chikusei_128/ --bands 128 --size 256 \
    --mask_path ./mask_256_128.mat --ckpt_dir ./Checkpoint_chik_v7nb --out_dir ./Result_chik_v7nb --epochs 91-100

# FixedSigma 臂:
.venv/bin/python Train_nb.py --model HyMCNetL3_V7SNB --bc_mode s4 --sigma 8 ... (σ 以波段计;128 波段 ≈ 5.1 nm/波段,σ=8 ≈ 41 nm)
```

## 5. 需要留意的设计点

1. **CASSI 前向模型**:沿用 `len_shift=2`,128 波段的测量宽度 = 128 + 127×2 = 382 像素(测试 256 patch 时 510)。真实 CASSI 很少做 128 波段的色散,这是仿真设定;若要更"物理",`--len_shift 1` 也可以(Dataset 与模型同一参数)。
2. **测试集只有 21 个 256² tile、同一航带**,场景多样性远低于 CAVE 的 10 景;结论要配合频段分析 / SAM 看。
3. **数据量**:训练区只有 35 个 256² tile(≈ 1/90 的 CAVE 像素量),大模型(46 M)会过拟合,建议先跑 `--mult 1,1,2`,并对比 train/test PSNR 差距。
4. **深层通道≠物理波段**仍然成立(与 V7 相同的 caveat),但第一层序列长度 128 已足以检验"长光谱序列是否让 SSM 占优"。

## 6. 其他 100+ 波段数据(备选,未下载)

| 数据 | 波段 | 规模 | 获取 |
|---|---|---|---|
| **ICVL 原始(519 波段,400–1000 nm,1.25 nm)** | 519,可重采样到 100–130 | 202 景自然图像,1392×1300,每景 1.9 GB,共 376 GB | HF `ICVL-BGU/ICVL_HS_2016`(gated),**本机 HF 登录已有访问权限**;挑 30–40 景约 70 GB,磁盘余 260 GB。最接近 CAVE 的"自然场景 + 100+ 波段"选项 |
| HyTexiLa(186 波段,405–996 nm) | 186 | 112 幅 1024² 材质纹理,ENVI float32,每幅 780 MB | `ftp://color-data.univ-lille.fr/datasets/hytexila/ENVI/`,匿名 FTP 可直接取 |
| Pavia Centre / Botswana / KSC 等 | 102–176 | 单景遥感 | ehu.eus 镜像对本机返回 403,需另找镜像 |

如果 Chikusei 上光谱扫描有效,下一步建议用 ICVL 原始数据重采样到 128 波段复现(自然场景、与 CAVE 同类),
`prepare_chikusei.py` 的分块读取 + tile 切分逻辑可以直接改写成 ENVI `.raw` 的读取。
