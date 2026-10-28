# 实验总览（截至 2026-10-28）

所有实验：CAVE 205 景训练（5000 样本/epoch，256×256 随机裁剪 + 翻转旋转 + 四拼接），10 场景测试；3-stage 展开网络；
Adam 4e-4，余弦到 1e-6，RMSE 多阶段损失（权重 1.3 / 0.7 / 0.5），300 epoch，batch 2，seed 42。
指标：Test_v3.py 的 PSNR / SSIM / SAM（逐场景平均，取 291-300 轮里测试 PSNR 最高的一轮）。

## 1. 版本谱系

| 版本 | 模型类（文件） | 训练脚本 | 检查点 | 结果目录 | 要点 | PSNR |
|---|---|---|---|---|---|---|
| SSR-S / M / L | `Model.Net`（`Model.py`） | 上游 | `Checkpoint_pretrain/model_{S,M,L}.pkl` | `Result/`、`Label/` | 原版，3 / 6 / 9 stage | 38.93 / 39.94 / 40.47 |
| 早期 Mamba S / M（2026-06） | `Mamba_Model.HyMCNetL3`（3 / 6 stage） | — | `Checkpoint_pretrain/model_mamba_{S,M}.pkl`（不入库） | `Result/*_mamba_*.mat` | 论文图 `make_figure.py` 用的版本 | — |
| 4stage | `Mamba_Model.HyMCNetL3`，stage 4 | `Train.py`（旧配置） | `Checkpoint_4stage/`（326 轮） | — | 早期实验 | 训练 43.28 |
| Parallel | `Mamba_Model.HyMCNetL3Parallel` | `Train_parallel.py` | `Checkpoint_parallel/` | `Result_parallel/`（仅 .mat） | 块内 Mamba 与轻量 CNN 并联 | 训练 43.20 |
| **HyMCNetL3（dwt）** | `Mamba_Model.HyMCNetL3` | `Train.py` | `Checkpoint_dwt/` | — | MamConvB 块 + Haar 小波上下采样 | **39.12** |
| V2 | `Mamba_Model.HyMCNetL3_V2` | `Train_v2.py` | `Checkpoint_v2/`（1 轮） | — | Haar 低频 Mamba / 高频 flow；跑 1 轮后放弃 | — |
| V3 | `Mamba_Model.HyMCNetL3_V3` | `Train_v3.py` | `Checkpoint_v3/` | `Result_v3/` | 块内分解，无满分辨率块 | 36.55 |
| V4 | `model_v4.HyMCNetL3_V4` | `Train_v4.py` | `Checkpoint_v4/` | `Result_v4/` | SSR 骨架 + 块内 Haar 双支路（LL→Mamba，HF→共享 ConvB） | 38.21 |
| V5 | `model_v5.HyMCNetL3_V5` | `Train_v5.py` | `Checkpoint_v5/` | `Result_v5/` | V4 + 高频各自 ConvB + SSR 初始化 + 修正 FFN 双重残差 | 38.51 |
| V6 | `model_v6.HyMCNetL3_V6` | `Train_v6.py` | `Checkpoint_v6/` | `Result_v6/` | V5 + 2×2 结构 token + Hilbert 扫描 + Mamba 内 2D 深度卷积 | 38.64 |
| Base | `model_baseline.HyMCNetL3_Base` | `Train_base.py` | `Checkpoint_base/` | `Result_base/` | SSR 步长卷积采样 + MamConvB 块（隔离采样层） | 38.75 |
| **V7** | `model_v7.HyMCNetL3_V7` | `Train_v7.py` / `run_v7.sh` | `Checkpoint_v7/`（229 / 300） | 待测 | HyMCNetL3 骨架，Mamba 改为 8×8 窗口内沿通道扫描 | 训练中 |
| V7S | `model_v7s.HyMCNetL3_V7S` | `Train_v7s.py --bc_mode {selective,s4,const}` | `Checkpoint_v7s_<mode>_sigma<σ>/` | — | V7 的 Mamba 换成 FixedSigmaMamba（A、dt 固定） | 未训练 |
| NB / V7NB / V7SNB | `model_nb.*` | `Train_nb.py` / `run_chik.sh` | `Checkpoint_chik_*/` | `Result_chik_*/` | 任意波段数版本，128 波段 Chikusei | 未训练 |
| SpecMamba on SSR-Net | `SpectralMambaNative/Model_SpecMamba.Net` | `SpectralMambaNative/Train_SpecMamba.py` | `SpectralMambaNative/Checkpoint_*` | — | 旧消融线：FixedSigmaMamba 插进原版 SSR 的 SRB；已不是主线 | 未训练 |

训练 PSNR（最终轮）：HyMCNetL3 43.39、Parallel 43.20、Base 42.70、V6 42.57、V5 42.47、V4 42.11、V3 39.49、V7 42.97（229 轮）。
所有 Mamba 版本训练 PSNR 都高于 SSR-S（41.04），测试却不占优，训练—测试差距（3~4 dB）大于 SSR（2.1 dB），问题来自结构而非训练不足。

## 2. 消融链（控制变量）

| 对比 | 唯一变化 | ΔPSNR |
|---|---|---|
| SSR-S → Base | SSRB 块换成 MamConvB 块，骨架不变 | −0.18 |
| Base → **HyMCNetL3** | 步长卷积采样换成 Haar 小波采样，块不变 | **+0.37** |
| Base → V5 | 块内改为小波双支路 | −0.24 |
| Base → V6 | 块内小波双支路 + 三项编码改进 | −0.11 |
| HyMCNetL3 → V7 | Mamba 扫描轴：空间 → 光谱 | 待定 |

## 3. 诊断结论

工具：`analyze_freq.py`（两级 Haar 子带 + FFT 径向分环的误差能量与频段 SNR，结果在 `Result_freq2/freq_analysis.csv`）和 `swap_bands.py`（两模型重建的低频 / 细节子带交叉重组）。

1. 所有模型误差以低频为主（LL2 占误差能量约 42%，FFT 低频环约 75%），决定 PSNR 的是低频结构。
2. HyMCNetL3 是唯一在最低频上赢 SSR-S 的模型（LL2 +0.24 dB），高频略输（约 −0.1 dB）；它相对 SSR-S 的 +0.19 dB 全部来自 64×64 以下的低频（只拿 HyMCNetL3 的 64×64 低频配 SSR-S 的高频就能复现 39.13）。
3. 频段 SNR 顺序 HyMCNetL3 > Base > V6 > V5：分频越深入块内，所有频段越差，不只是高频或低频的问题。
4. V6 的三项编码改动没有让块内低频 Mamba 学到更多结构（16×16 尺度上「V5 低频 + V6 高频」不低于 V6 自身），V6 比 V5 多的 0.13 dB 来自细节子带。
5. 发现的 bug：`Mamba_Model.FFN` 自带残差，块外又加一次，使 FFN 支路实际为 `x + LN(x) + ffn(LN(x))`。V5 起改用 `Model.FFN`；HyMCNetL3 和 V7 仍沿用旧 FFN，修正后是否有增益尚未验证。

细节和表格：`docs/weekly/周报_2026-10-15.md`、`docs/weekly/周报_2026-10-19.md`。

## 4. V7：沿光谱维扫描（主线）

- 动机：CASSI 的核心耦合在光谱维；空间 1D 扫描在二维图像上结构性受限，而每个像素的波段序列天然有序。
- 设计：`model_v7.py` 相对 HyMCNetL3 只改一处，每个 MambaB 里的 `Mamba_Att`（H×W 光栅扫描）换成 `SpecMamba_Att`：特征图切成 8×8 窗口，每个窗口是一条以通道为序列、每个通道 token 为 64 个像素的序列，Mamba 沿通道扫描。空间结构交给 5×5 位置卷积、11×11 ConvB 和 U-Net 多尺度路径。
- 已验证：光谱因果性成立（扰动第 40 个及以后通道，前 40 个输出不变）；窗口间扫描互不影响；参数 2.95M（HyMCNetL3 2.94M）；一步 0.27 s、峰值显存 10.6 GB。
- 风险：深层通道（56 / 112）不再对应物理波段，「光谱序」只在第一层严格成立；跨窗口信息完全依赖卷积。
- V7 训练于 2026-10-19 启动，机器 2026-10-28 重启后由 `run_v7.sh` 续训，当前 229 / 300。

V7S（`model_v7s.py`，`SpectralMambaNative/README.md`）：把 Mamba 的时间轴选择性（可学 A、随输入的 dt）换成固定光谱先验 `A[d,n] = -(n+1)/σ`、`dt = 1 波段`，B/C 三档：`selective`（随输入）、`s4`（每通道可学常量，LTI）、`const`（B=C=1）。V7 原生是第四臂。四臂参数都在 2.84~2.95M，显存 9.5~9.9 GB，冒烟通过。

## 5. 128 波段线（Chikusei）

CAVE 只有 28 波段，光谱序列太短。`model_nb.py` 把 HyMCNetL3 / V7 / V7S 泛化到任意波段数（28 波段下与原模型逐位一致，已验证），在 Chikusei 128 波段（35 个 256² 训练 tile，21 个 256² 测试 tile）上比较空间扫描与光谱扫描。显存限制下训练 patch 用 128×128，`--mult 1,1,2`（15M）是推荐折中。详见 `docs/100bands.md`。

## 6. 待办（按优先级）

1. V7 训完后：`Test_v3.py --model HyMCNetL3_V7 --epochs 291-300`，再做 `analyze_freq.py` 与 `swap_bands.py`（对 HyMCNetL3），看光谱扫描是否改善低频。
2. 若 V7 ≥ HyMCNetL3：训练 V7S 三档（σ=4 起，再扫 2 / 8），然后启动 Chikusei 线（`run_chik.sh`，先 V7NB 再 NB 对照）。
   若 V7 < HyMCNetL3：「Mamba 学全局」的收益来自空间扫描，主线改为强调「小波采样 + Mamba」，补充消融；Chikusei 线仍可作为光谱扫描的验证。
3. 修正 HyMCNetL3 的 FFN 双重残差，单独训练一次，量化影响（可能是廉价增益）。
4. 论文指标用 MATLAB `cal_psnr_ssim.m` 复核 SSIM；测试集只有 10 景，0.1~0.2 dB 的差异要谨慎解读。
5. 若 Chikusei 上光谱扫描有效，用 ICVL 原始数据（519 波段）重采样到 128 波段复现（自然场景，与 CAVE 同类）。
