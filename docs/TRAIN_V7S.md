# V7S 训练手册（在新服务器上执行）

这份文档写给在新机器上接手训练的人或 Claude 会话。目标：**训练 V7S 三档消融（selective / s4 / const，σ=4），测试，并与 V7、HyMCNetL3 对比**。
全部命令在仓库根目录、`.venv` 激活后执行。看完 §0 就可以开始。

## 0. 一页纸总结

| 项 | 值 |
|---|---|
| 模型 | `model_v7s.HyMCNetL3_V7S`：V7 骨架，Mamba 换成 `FixedSigmaMamba`（A、dt 固定为光谱先验） |
| 三臂 | `--bc_mode selective`（B/C 随输入）、`s4`（B/C 每通道可学常量，LTI）、`const`（B=C=1） |
| 对照 | V7 原生（A 可学、dt 随输入）38.86 dB；HyMCNetL3（空间扫描）39.12 dB；SSR-S 38.93 dB |
| 训练脚本 | `Train_v7s.py`，或队列 `run_v7s.sh`（默认依次跑三臂） |
| 配方 | 与 V7 完全相同：CAVE 205 景，5000 样本/epoch，256² patch，batch 2，Adam 4e-4 余弦到 1e-6，300 epoch，seed 42 |
| 每臂资源 | 显存 9.5~9.8 GB（batch 2）；**内存 24 GB**（CAVE 全部常驻 float32）；RTX 4090 每 epoch 约 580 s，300 epoch ≈ 48 h |
| 检查点 | `Checkpoint_v7s_<mode>_sigma4/model_XXX.pkl`，每 epoch 一个（含优化器，约 36 MB），自动续训 |
| 测试 | `Test_v3.py --model HyMCNetL3_V7S --bc_mode <mode> --sigma 4 --ckpt_dir ... --epochs 291-300` |

启动（单卡，三臂串行约 6 天）：

```bash
tmux new -s v7s -d 'bash run_v7s.sh'
tail -f training_v7s_selective_sigma4.log
```

多卡（每臂一张卡，约 2 天）：

```bash
tmux new -s v7s_sel   -d 'GPU=0 ARMS=selective bash run_v7s.sh'
tmux new -s v7s_s4    -d 'GPU=1 ARMS=s4        bash run_v7s.sh'
tmux new -s v7s_const -d 'GPU=2 ARMS=const     bash run_v7s.sh'
```

三臂并行需要 **3 × 24 GB ≈ 72 GB 内存**；内存不够就串行或两两并行。

## 1. 开始前检查

### 1.1 代码与数据

```bash
git clone git@github.com:XingyZhou/SSR.git && cd SSR      # 或 https://github.com/XingyZhou/SSR.git
```

需要从旧机器 mlp4（`/home/mlp4/data/ZHOU/SSR`）复制的东西，用仓库里的脚本（在 **mlp4** 上执行）：

```bash
bash scripts/pack_transfer.sh user@newhost:/path/to/SSR
```

本任务最少需要：

| 路径 | 用途 | 大小 |
|---|---|---|
| `CAVE_1024_28/scene*.mat`（205 个） | 训练集 | 9.8 GB |
| `Test_data/scene01..10.mat` | 测试集 | 63 MB |
| `mask_256_28.mat` | 掩膜（已在 git 里） | 5 MB |
| `Checkpoint_v7/model_300.pkl`、`Checkpoint_dwt/model_300.pkl`、`Checkpoint_pretrain/model_S.pkl` | 对比用：V7、HyMCNetL3、SSR-S | 36 + 12 + 7 MB |

核对：

```bash
ls CAVE_1024_28/*.mat | wc -l      # 205
ls Test_data/*.mat | wc -l         # 10
head -1 CAVE_1024_28/train_list.txt  # ./CAVE_1024_28/scene1.mat （相对路径，所以必须在仓库根目录运行）
```

### 1.2 环境

按 `docs/SETUP.md` §2 建 `.venv`（Python 3.11，torch 2.8.0，mamba-ssm 2.2.5）。然后：

```bash
source .venv/bin/activate
python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available(), mamba_ssm.__version__)"
python SpectralMambaNative/test_spectral_mamba_native.py     # FixedSigmaMamba 三档单元测试，须全部 PASS
nvidia-smi                                                    # 每臂要 10 GB 空闲显存
free -g                                                       # 每臂要 24 GB 空闲内存
```

`causal_conv1d` 装不装都行：V7 是在没装的环境里训的（Mamba 走 torch conv1d + `selective_scan_fn`），**三臂和 V7 要在同一种环境下跑**，否则对比不公平。新机器上如果装了 `causal_conv1d`，就都装着；不装就都不装。

### 1.3 整网冒烟（1 分钟）

用两个测试场景走一遍真实的数据管线（和训练同样的 batch 2、256²），三档各做一步前向 + 反向，并检查检查点能按同一 `bc_mode` 重新加载：

```bash
python - <<'EOF'
import argparse, torch, numpy as np, scipy.io as sio, torch.utils.data as tud
from Utils import *
from Dataset import dataset
from model_v7s import HyMCNetL3_V7S
files = sorted(loadpath('./Test_data/test_list.txt'))[:2]
HSI = np.zeros((256, 256, 28, 2))
for i, f in enumerate(files): HSI[:, :, :, i] = sio.loadmat(f)['img']
base = dict(data_path='./Test_data/', mask_path='./mask_256_28.mat', size=256, stage=3, bands=28, len_shift=2,
            isTrain=True, trainset_num=2, testset_num=2, scene_num=2, seed=42, batch_size=2)
opt = argparse.Namespace(**base, bc_mode='selective', sigma=4.0, dt_fixed=1.0)
g, label, Phi, Phi_s = next(iter(tud.DataLoader(dataset(opt, HSI), batch_size=2, shuffle=False)))
for mode in ['selective', 's4', 'const']:
    torch.cuda.reset_peak_memory_stats()
    m = HyMCNetL3_V7S(argparse.Namespace(**base, bc_mode=mode, sigma=4.0, dt_fixed=1.0)).cuda()
    out = m(g=g.cuda(), input_mask=(Phi.cuda(), Phi_s.cuda()))
    loss = sum(torch.sqrt(torch.nn.functional.mse_loss(o, label.cuda())) for o in out); loss.backward()
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    ok = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    print(f"{mode:9s} trainable {n_tr:.3f}M  out {tuple(out[-1].shape)}  grads_finite {ok}  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    HyMCNetL3_V7S(argparse.Namespace(**base, bc_mode=mode, sigma=4.0, dt_fixed=1.0)).load_state_dict(sd); print("   reload OK")
    del m, out, loss; torch.cuda.empty_cache()
EOF
```

在 mlp4 的 4090 上的输出（2026-10-29 验证）：

```
selective trainable 2.912M  out (2, 28, 256, 256)  grads_finite True  peak 9.9 GB
s4        trainable 2.904M  out (2, 28, 256, 256)  grads_finite True  peak 9.5 GB
const     trainable 2.843M  out (2, 28, 256, 256)  grads_finite True  peak 9.5 GB
```

参数量对不上、显存爆、出现 NaN，先别开训练。

## 2. 训练

### 2.1 队列脚本 `run_v7s.sh`

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ARMS` | `selective s4 const` | 要跑的臂，空格分隔，按顺序 |
| `SIGMA` | `4` | 光谱先验宽度（以波段计） |
| `GPU` | `0` | 用哪张卡（写进 `CUDA_VISIBLE_DEVICES`） |
| `EPOCHS` | `300` | 到第几轮停（余弦调度固定按 300 轮） |
| `PY` | `.venv/bin/python` | 解释器 |
| `SCT_KEY` / `BARK_KEY` | 空 | 设了才启动 `monitor_train.py`（微信 Server酱 / Bark 推送，每 2 h 一次，异常即时告警） |

产物：`Checkpoint_v7s_<mode>_sigma<σ>/`、`training_v7s_<mode>_sigma<σ>.log`、（可选）`monitor_v7s_<mode>_sigma<σ>.log`。

### 2.2 手动等价命令

```bash
CUDA_VISIBLE_DEVICES=0 python -u Train_v7s.py --bc_mode s4 --sigma 4 2>&1 | tee -a training_v7s_s4_sigma4.log
```

其他参数：`--dt_fixed 1.0`（固定 dt = 1 个波段间距，不要改）、`--ckpt_dir`（默认 `./Checkpoint_v7s_<mode>_sigma<σ>`）、`--max_epochs`。
不要传 `--stage`（它是 str 类型的历史遗留，默认 3 是 int，传了会出错）。

### 2.3 续训 / 中断

- 进程被杀、机器重启：原命令重新执行即可，脚本自动找 `ckpt_dir` 里编号最大的 `model_XXX.pkl`，连优化器状态和学习率调度一起恢复。
- 想提前停：`tmux kill-session -t v7s`，之后用同样命令继续。
- 日志每 50 步一行 `epoch step/2500 loss`，每 epoch 结束一行 `epoch = N , loss = ... , Avg PSNR = ... ,time = ... s`。

### 2.4 健康判断

| 现象 | 正常值（V7 原生作参考） |
|---|---|
| epoch 1 结束时训练 PSNR | 30~33 dB |
| epoch 50 | 约 39 dB |
| epoch 150 | 约 41.5 dB |
| epoch 300 | V7 为 43.20 dB；V7S 三档预期略低，const 最低 |
| 每 epoch 时间 | 4090 约 580 s；若明显更慢，看是不是和别的任务共用了卡 |
| loss | 单调缓降到约 0.0107；出现 NaN 立即停 |

训练 PSNR 高不代表测试好（所有 Mamba 版本训练—测试差距 3~4 dB），**只看最终测试**。

## 3. 测试

每臂训完（或拿到 `model_300.pkl`）后：

```bash
for mode in selective s4 const; do
  python Test_v3.py --model HyMCNetL3_V7S --bc_mode $mode --sigma 4 \
      --ckpt_dir ./Checkpoint_v7s_${mode}_sigma4 --out_dir ./Result_v7s_${mode}_sigma4 --epochs 291-300
done
```

`--bc_mode / --sigma` 必须和训练时一致，否则 `load_state_dict` 报键不匹配（s4 多出可学 B/C，const 把 B/C 变成 buffer）。
输出在 `--out_dir`：`report.md`（最佳轮的 PSNR / SSIM / SAM、逐景）、`summary.csv`（10 轮）、`per_scene.csv`、`best.json`、`recon/`（最佳轮重建 .mat）。

测试显存约 3 GB，可以在训练同时跑。

## 4. 对比与诊断

### 4.1 主表

| 模型 | 参数 | PSNR | SSIM | SAM | 备注 |
|---|---|---|---|---|---|
| SSR-S | 1.81M | 38.93 | 0.9731 | 4.20 | 原版 Transformer |
| HyMCNetL3 | 2.94M | **39.12** | 0.9742 | 3.78 | 小波采样 + 空间扫描 Mamba |
| V7 | 2.95M | 38.86 | 0.9728 | 3.96 | 光谱扫描，原生 S6（A 可学、dt 随输入） |
| V7S selective | 2.91M | | | | 本次 |
| V7S s4 | 2.90M | | | | 本次 |
| V7S const | 2.84M | | | | 本次 |

### 4.2 频段诊断（V7 / HyMCNetL3 用的同一套工具）

模型写法 `名字:模块.类:权重[:bc_mode=xx,sigma=4]`，第四段只有 V7S 需要：

```bash
python analyze_freq.py --out_dir ./Result_freq_v7s --models \
    SSR-S:Model.Net:./Checkpoint_pretrain/model_S.pkl \
    HyMCNetL3:Mamba_Model.HyMCNetL3:./Checkpoint_dwt/model_300.pkl \
    V7:model_v7.HyMCNetL3_V7:./Checkpoint_v7/model_300.pkl \
    V7S-sel:model_v7s.HyMCNetL3_V7S:./Checkpoint_v7s_selective_sigma4/model_300.pkl:bc_mode=selective,sigma=4 \
    V7S-s4:model_v7s.HyMCNetL3_V7S:./Checkpoint_v7s_s4_sigma4/model_300.pkl:bc_mode=s4,sigma=4 \
    V7S-const:model_v7s.HyMCNetL3_V7S:./Checkpoint_v7s_const_sigma4/model_300.pkl:bc_mode=const,sigma=4

python swap_bands.py --levels 1 2 3 4 \
    --a V7S-s4:model_v7s.HyMCNetL3_V7S:./Checkpoint_v7s_s4_sigma4/model_300.pkl:bc_mode=s4,sigma=4 \
    --b V7:model_v7.HyMCNetL3_V7:./Checkpoint_v7/model_300.pkl
```

看三件事：(1) LL2 频段 SNR 是否高于 V7 的 25.57；(2) 子带互换里「V7S 低频 + V7 高频」是否同时超过两者；(3) 中频（HL2 / LH2）是不是仍然是短板（V7 的 LH2 比 HyMCNetL3 差 0.43 dB）。
已有结果：`Result_freq3/freq_analysis.csv`（SSR-S / HyMCNetL3 / Base / V7）、`Result_freq3/swap_bands.txt`。

### 4.3 结果怎么读

- **V7S-selective ≥ V7**：固定光谱先验有效，"把时间轴选择性换成物理先验、保留内容轴选择性"成立，可作主张。接着扫 σ（`SIGMA=2` / `SIGMA=8`，只跑最好的那一档）。
- **s4 ≈ selective**：内容轴选择性对光谱也没用，模型退回 LTI；论述口径见 `SpectralMambaNative/README.md` §3.4。
- **const 明显最差**：说明可学的 B/C 至少有一个在起作用（预期如此，这是下界）。
- **三档都 < V7**：光谱先验在 28 波段上无增益；结合 V7 < HyMCNetL3，28 波段光谱扫描这条线到此为止，把精力转向 128 波段 Chikusei（`docs/100bands.md`，V7SNB 已就绪）。

## 5. 完成后

1. 把 `Result_v7s_*/`（csv / md / json）和 `Result_freq_v7s/` 提交：`git add -A && git commit -m "V7S sigma4 results" && git push`（.gitignore 已排除 pkl / mat / log）。
2. 在 `docs/EXPERIMENTS.md` §1 表格里填上三臂的 PSNR，§2 消融链加一行「V7 → V7S」，§6 待办更新。
3. 若需要把权重带回 mlp4：只拷每臂的 `model_300.pkl`（36 MB）即可。

## 6. 已知坑

- `Train_v7s.py` / `Test_v3.py` / `analyze_freq.py` / `swap_bands.py` 内部默认 `CUDA_VISIBLE_DEVICES=0`，外部设了环境变量才会覆盖（`GPU=1 bash run_v7s.sh` 或 `CUDA_VISIBLE_DEVICES=1 python ...`）。
- `Utils.py` 关闭了 cudnn（`torch.backends.cudnn.enabled = False`）以保证可复现，别改。
- 第一层（dim 28）通道才是物理波段；56 / 112 层的「光谱序」是学到的通道序，σ 在那里只是通道平滑先验。写结论时注意这一点。
- 测试集只有 10 景，0.1~0.2 dB 的差异要谨慎解读；最终论文指标用 MATLAB `cal_psnr_ssim.m` 复核 SSIM。
