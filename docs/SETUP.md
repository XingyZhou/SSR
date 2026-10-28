# 新服务器安装与迁移

目标：在另一台机器上把仓库跑起来，并接着现在的实验做。代码从 GitHub 取，数据和权重从旧机器（mlp4，`/home/mlp4/data/ZHOU/SSR`）rsync。

## 1. 代码

```bash
git clone https://github.com/XingyZhou/SSR.git
cd SSR
```

## 2. Python 环境（uv）

原环境：Python 3.11.7、torch 2.8.0+cu128、mamba-ssm 2.2.5（本地编译）、无 causal_conv1d。版本见 `requirements.txt`。

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

# 1) torch：选和驱动匹配的 CUDA 轮子（nvidia-smi 右上角 CUDA Version >= 轮子版本即可）
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 2) 其余依赖
uv pip install einops==0.8.1 timm==1.0.22 numpy==2.2.6 scipy==1.15.3 h5py==3.16.0 pandas==2.3.1 \
    scikit-image==0.25.2 matplotlib==3.10.9 thop fvcore "transformers==4.55.0" ninja packaging

# 3) mamba-ssm：二选一
#    a. 预编译轮子（最省事）。到 https://github.com/state-spaces/mamba/releases 找 2.2.5 里
#       与 torch 2.8 / cu12 / cp311 匹配的 whl（文件名形如 mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl），
#       cxx11abi 取 python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)" 的值
uv pip install --no-deps <该 whl 的 URL>
#    b. 本地编译（需要 nvcc，且 nvcc 的 CUDA 主版本与 torch 一致，约 10-20 分钟）
uv pip install --no-build-isolation mamba-ssm==2.2.5
```

验证：

```bash
python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available(), mamba_ssm.__version__)"
python SpectralMambaNative/test_spectral_mamba_native.py     # FixedSigmaMamba 三档单元测试，应全部 PASS
python Cal_para_flops.py                                      # HyMCNetL3 参数量 2.94M
```

注意：
- `Utils.py` 里有 `torch.backends.cudnn.enabled = False`，这是为了结果可复现，不要删。
- `causal_conv1d` 没装时 Mamba 走 torch conv1d + `selective_scan_fn` 慢路径；V7 与 V7S 都走同一路径，公平。装了会更快，但要同一台机器上所有臂保持一致。
- 旧机器上另有一个 anaconda python（torch 2.0.1+cu118），**不是**训练环境，不要照抄。

## 3. 数据

| 数据 | 放哪 | 来源 |
|---|---|---|
| CAVE 训练集（205 景 1024×1024×28，9.8 GB） | `CAVE_1024_28/scene*.mat` | MST 系列的百度盘（见 README「上游 SSR」），或从旧机器 rsync |
| 测试集（10 景 256×256×28） | `Test_data/scene01..10.mat` | 同上 |
| 掩膜 | `mask_256_28.mat`（已在 git 里）、`mask_256_128.mat`、`mask_128_128.mat` | 128 版由 `prepare_chikusei.py` 生成，或 rsync |
| Chikusei 128 波段 | `Chikusei_128/{train,test}/` | 由 `HSI_raw/chikusei/` 原始立方体用 `python prepare_chikusei.py --train_size 256` 生成（见 `docs/100bands.md`）。原始数据 1.4 GB zip：Naoto Yokoya 主页 https://naotoyokoya.com/Download.html ，CC BY 3.0，论文须致谢 SAL/U-Tokyo |

`train_list.txt` / `test_list.txt` 里都是相对路径 `./CAVE_1024_28/...`，所以训练脚本必须在仓库根目录运行。

## 4. 权重、结果、日志：从旧机器复制

GitHub 上只有代码和 csv/md 结果。其余用仓库里的脚本 rsync（可断点续传）：

```bash
# 在旧机器 mlp4 上执行；先看清单和大小，再加目标地址
bash scripts/pack_transfer.sh
bash scripts/pack_transfer.sh user@newhost:/path/to/SSR          # 约 23 GB：数据 + 掩膜 + 各版本最终权重 + V7 全部权重 + 重建结果 + 日志
FULL=1 bash scripts/pack_transfer.sh user@newhost:/path/to/SSR   # 再加上所有 per-epoch 检查点（约 65 GB），一般不需要
```

要续训 V7 只需要 `Checkpoint_v7/` 里最新的一个 `model_XXX.pkl`：`Train_v7.py` 启动时自动找最大编号续训。

## 5. 训练与监控

训练脚本把检查点目录写死在脚本里（`Train_v7.py -> ./Checkpoint_v7`，`Train.py -> ./Checkpoint_dwt`，`Train_nb.py` 用 `--ckpt_dir`），每个 epoch 存一个 `model_XXX.pkl`（约 12 MB × 300 轮 = 3.6 GB，V7 为 36 MB × 300）。

```bash
# 直接跑
python Train_v7.py 2>&1 | tee -a training_v7.log

# tmux + 微信推送（Server酱 Turbo SendKey）；run_*.sh 里第一行 cd 是旧机器的绝对路径，换机器后改成新路径
export SCT_KEY=SCTxxxx
tmux new -s train_v7 -d 'bash run_v7.sh'
tail -f training_v7.log
```

`monitor_train.py` 每 2 小时推送一次进度，训练进程消失、loss 变 NaN 或日志 15 分钟不更新时立即告警。
环境变量：`TRAIN_LOG`、`TRAIN_PROC`、`TOTAL_EPOCHS`、`SCT_KEY`、`BARK_KEY`。

显存参考（RTX 4090）：HyMCNetL3 / V7 在 batch 2、256×256 下约 10 GB，一个 epoch（5000 样本）约 10 分钟，300 epoch 约 2 天；
128 波段线每个 epoch 40-50 分钟，见 `docs/100bands.md`。旧机器内存 31 GB，V7 训练常驻 24 GB（整个 CAVE 常驻内存），内存小的机器要注意。

## 6. 测试

```bash
python Test_v3.py --model HyMCNetL3_V7 --ckpt_dir ./Checkpoint_v7 --out_dir ./Result_v7 --epochs 291-300
```

`--model` 可用的类名：`HyMCNetL3`、`HyMCNetL3_Base`、`HyMCNetL3_V3`…`HyMCNetL3_V7`、`HyMCNetL3_V7S`（后者再加 `--bc_mode --sigma`）。
128 波段用 `Test_nb.py`。论文指标最终用 MATLAB `cal_psnr_ssim.m` 复核 SSIM。
