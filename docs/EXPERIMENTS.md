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
| **V7** | `model_v7.HyMCNetL3_V7` | `Train_v7.py` / `run_v7.sh` | `Checkpoint_v7/`（300） | `Result_v7/` | HyMCNetL3 骨架，Mamba 改为 8×8 窗口内沿通道扫描 | **38.86** |
| V7S | `model_v7s.HyMCNetL3_V7S` | `Train_v7s.py --bc_mode {selective,s4,const}` | `Checkpoint_v7s_<mode>_sigma<σ>/` | — | V7 的 Mamba 换成 FixedSigmaMamba（A、dt 固定） | 未训练 |
| NB / V7NB / V7SNB | `model_nb.*` | `Train_nb.py` / `run_chik.sh` | `Checkpoint_chik_*/` | `Result_chik_*/` | 任意波段数版本，128 波段 Chikusei | 未训练 |
| SpecMamba on SSR-Net | `SpectralMambaNative/Model_SpecMamba.Net` | `SpectralMambaNative/Train_SpecMamba.py` | `SpectralMambaNative/Checkpoint_*` | — | 旧消融线：FixedSigmaMamba 插进原版 SSR 的 SRB；已不是主线 | 未训练 |

训练 PSNR（最终轮）：HyMCNetL3 43.39、Parallel 43.20、V7 43.20、Base 42.70、V6 42.57、V5 42.47、V4 42.11、V3 39.49。
所有 Mamba 版本训练 PSNR 都高于 SSR-S（41.04），测试却不占优，训练—测试差距（3~4 dB）大于 SSR（2.1 dB），问题来自结构而非训练不足。

完整指标（各自最佳轮；推理时间为 Test_v3.py 单景 256×256 中位数，同一张 4090）：

| 模型 | 参数 | PSNR | SSIM | SAM (°) | stage1 / stage2 PSNR | 训练 PSNR | ms/景 |
|---|---|---|---|---|---|---|---|
| SSR-S | 1.81M | 38.93 | 0.9731 | 4.20 | — | 41.04 | — |
| HyMCNetL3 | 2.94M | **39.12** | **0.9742** | **3.78** | — | 43.39 | — |
| V7 | 2.95M | 38.86 | 0.9728 | 3.96 | 35.94 / 37.91 | 43.20 | 26.4 |
| Base | 2.04M | 38.75 | 0.9724 | 3.99 | 35.66 / 38.17 | 42.63 | 21.0 |
| V6 | 3.15M | 38.64 | 0.9712 | 4.22 | 35.33 / 37.58 | 42.48 | 26.6 |
| V5 | 2.90M | 38.51 | 0.9706 | 4.15 | 35.41 / 37.51 | 42.47 | — |
| V4 | 2.04M | 38.21 | 0.9690 | 4.40 | 34.72 / 36.90 | 42.02 | — |

## 2. 消融链（控制变量）

| 对比 | 唯一变化 | ΔPSNR |
|---|---|---|
| SSR-S → Base | SSRB 块换成 MamConvB 块，骨架不变 | −0.18 |
| Base → **HyMCNetL3** | 步长卷积采样换成 Haar 小波采样，块不变 | **+0.37** |
| Base → V5 | 块内改为小波双支路 | −0.24 |
| Base → V6 | 块内小波双支路 + 三项编码改进 | −0.11 |
| HyMCNetL3 → **V7** | Mamba 扫描轴：空间（H×W 光栅）→ 光谱（8×8 窗口内沿通道） | **−0.26** |

读法：采样层用小波是正收益；块内分频始终是负收益；把扫描轴换到光谱维在 28 波段上也是负收益。目前唯一确认有效的 Mamba 用法是「小波采样骨架 + 沿空间扫描」。

## 3. 诊断结论

工具：`analyze_freq.py`（两级 Haar 子带 + FFT 径向分环的误差能量与频段 SNR）和 `swap_bands.py`（两模型重建的低频 / 细节子带交叉重组，oracle）。诊断统一用各模型 `model_300.pkl`。
结果：`Result_freq2/freq_analysis.csv`（SSR-S / HyMCNetL3 / Base / V5 / V6）、`Result_freq3/freq_analysis.csv` + `Result_freq3/swap_bands.txt`（SSR-S / HyMCNetL3 / Base / V7）。

1. 所有模型误差以低频为主（LL2 占误差能量约 41~43%，FFT 低频环约 75%），决定 PSNR 的是低频结构。
2. HyMCNetL3 是唯一在最低频上赢 SSR-S 的模型（LL2 +0.24 dB），高频略输（约 −0.1 dB）；它相对 SSR-S 的 +0.19 dB 全部来自 64×64 以下的低频（只拿 HyMCNetL3 的 64×64 低频配 SSR-S 的高频就能复现 39.13）。
3. 频段 SNR 顺序 HyMCNetL3 > Base > V6 > V5：分频越深入块内，所有频段越差，不只是高频或低频的问题。
4. V6 的三项编码改动没有让块内低频 Mamba 学到更多结构（16×16 尺度上「V5 低频 + V6 高频」不低于 V6 自身），V6 比 V5 多的 0.13 dB 来自细节子带。
5. **V7 的低频确实学到了，但不如空间扫描**：LL2 SNR 25.57，高于 Base（25.45）、低于 SSR-S（25.66）和 HyMCNetL3（25.90）。V7 低频 + SSR-S 高频在 64×64 以下尺度达 39.00，同时超过两者（V7 38.85、SSR-S 38.93），反向互换同时低于两者，所以 V7 的低频实打实优于 SSR-S，拖后腿的是高频细节。这个模式与 HyMCNetL3 相同，只是量更小（HyMCNetL3 低频 + SSR-S 高频 = 39.13）。
6. **V7 输给 HyMCNetL3 的 0.27 dB 落在 16×16 到 128×128 之间的中低频**，不在最低频也不在最细节：只保留 V7 的 16×16 低频、其余用 HyMCNetL3 可到 39.08（差 0.04）；只保留 V7 最细一层细节也只损失 0.05；两列互换曲线在 64×64 处交叉，每上一层尺度 V7 多亏 0.04~0.09 dB。V7 的细节子带全部劣于 HyMCNetL3，LH2 最差（−0.43 dB）。解释：8×8 窗口内扫描丢掉了空间扫描提供的中尺度空间上下文，跨窗口信息只靠 5×5 / 11×11 卷积与 Haar 多尺度路径传递。
7. 发现的 bug：`Mamba_Model.FFN` 自带残差，块外又加一次，使 FFN 支路实际为 `x + LN(x) + ffn(LN(x))`。V5 起改用 `Model.FFN`；HyMCNetL3 和 V7 仍沿用旧 FFN，修正后是否有增益尚未验证。

细节和表格：`docs/weekly/周报_2026-10-15.md`、`docs/weekly/周报_2026-10-19.md`、`docs/weekly/周报_2026-10-28.md`。

## 4. V6：用编码补救块内低频 Mamba（已结束，放弃）

- **问题假设**（来自 V5 诊断）：V5 的块内低频 Mamba 沿行优先 1D 因果扫描，上下相邻像素在序列上相距 128，看不到二维结构，所以低频学不好。
- **设计**（`model_v6.py`，相对 V5 只改低频支路的 Mamba 输入与扫描顺序，不增加扫描方向）：
  1. 结构 token：四个 Haar 子带拼接（4C）经 1×1 投影到 C，每个 token 编码 2×2 邻域的「均值 + 水平边 + 垂直边 + 对角」，Mamba 扫描结构单元而非孤立像素；输出只加回 LL。
  2. Hilbert 曲线扫描顺序：相邻 token 的二维距离恒为 1（行优先平均约 126），零参数、一次固定置换。
  3. Mamba 内部 1D 因果卷积换成 2D 深度卷积，token 在扫描前保留二维邻域；状态空间参数沿用 mamba_ssm 初始化。
- **实现验证**：参数 3.15M；块初始偏差 0.072~0.079，接近恒等映射；一步 0.57 s、显存 6.5 GB。训练 2026-10-15 ~ 10-17。
- **结果**：38.64 / 0.9712 / 4.22；相对 V5 +0.13，相对 Base −0.11，相对 HyMCNetL3 −0.48。
- **诊断**：把 V6 的 128×128 低频换成 HyMCNetL3 的涨 0.39 dB，只换高频涨 0.08，短板仍在低频；16×16 尺度「V5 低频 + V6 高频」（38.65）不低于 V6 自身（38.64），说明三项编码改动没有让低频 Mamba 学到更多结构，增益来自各级细节子带。
- **结论**：块内「低频走 Mamba、高频走卷积」两轮设计（V5、V6）都输给不分频的 Base，证据足够，放弃这条路线。

## 5. V7：沿光谱维扫描（已完成，结论为负）

- **动机**：CASSI 的核心耦合在光谱维；空间 1D 扫描在二维图像上结构性受限，而每个像素的波段序列天然有序。验证「Mamba 在 CASSI 里最合适的扫描轴」。
- **设计**：`model_v7.py` 相对 HyMCNetL3 只改一处，每个 MambaB 里的 `Mamba_Att`（H×W 光栅扫描）换成 `SpecMamba_Att`：特征图切成 8×8 窗口，每个窗口是一条以通道为序列、每个通道 token 为 64 个像素的序列（与 SSR 的 WSSA 光谱注意力同一 token 布局），Mamba 沿通道扫描。空间结构交给 5×5 位置卷积、11×11 ConvB 和 U-Net 多尺度路径。骨架、小波采样、损失、训练配置均不变，HyMCNetL3 与 V7 的对比隔离出「扫描轴」一个因素。
- **已验证**：光谱因果性成立（扰动第 40 个及以后通道，前 40 个输出不变）；窗口间扫描互不影响；参数 2.95M（HyMCNetL3 2.94M）；一步 0.27 s、峰值显存 10.6 GB。
- **训练过程**：2026-10-19 启动；10-21 03:21 机器死机，停机一周；10-28 03:07 重启后 `run_v7.sh` 从 model_227 续训，10-28 15:00 跑完 300 轮（每轮 578 s）。续训前后曲线平滑衔接。训练 PSNR 300 轮 43.20，最高 280 轮 43.29。
- **测试结果**（`Result_v7/`）：291-300 轮测试 PSNR 在 38.840~38.858 之间，已收敛；最佳 294 轮 38.858 / 0.9728 / 3.961，stage1 / 2 为 35.94 / 37.91。是 model_v 系列最好的一版（比 Base +0.11、V6 +0.22），但低于 HyMCNetL3（39.12）0.26 dB，也略低于 SSR-S（38.93）。逐场景相对 Base：scene09 +0.66、scene03 +0.47、scene02 +0.29，scene05 −0.21、scene01 −0.19，不是全面提升。训练—测试差距 4.3 dB，与 HyMCNetL3（4.27）相当，没有改善。
- **频段诊断**：见第 3 节第 5、6 条。一句话：低频学到了（优于 SSR-S 和 Base），但 16~128 尺度的中低频输给空间扫描；光谱建模本身没坏，坏的是窗口内扫描丢掉的空间上下文。
- **结论**：在 28 波段 CAVE 上，「Mamba 学全局低频」的收益来自空间扫描，光谱扫描不能替代它。主线改为强调「小波采样 + Mamba（空间扫描）」，V7 作为「扫描轴」消融保留。若要救 V7，方向是扩大窗口或空间 / 光谱两种扫描并联，而不是改光谱那一侧。
- **风险（已兑现一部分）**：深层通道（56 / 112）不再对应物理波段，「光谱序」只在第一层严格成立；跨窗口信息完全依赖卷积（诊断证实这是中尺度损失的来源）。

V7S（`model_v7s.py`，`SpectralMambaNative/README.md`）：把 Mamba 的时间轴选择性（可学 A、随输入的 dt）换成固定光谱先验 `A[d,n] = -(n+1)/σ`、`dt = 1 波段`，B/C 三档：`selective`（随输入）、`s4`（每通道可学常量，LTI）、`const`（B=C=1）。V7 原生是第四臂。四臂参数都在 2.84~2.95M，显存 9.5~9.9 GB，冒烟通过。V7 结论为负后，V7S 在 CAVE 上的优先级降低，只在 Chikusei 上光谱扫描有效时再训。

## 6. 128 波段线（Chikusei）

CAVE 只有 28 波段，光谱序列太短。`model_nb.py` 把 HyMCNetL3 / V7 / V7S 泛化到任意波段数（28 波段下与原模型逐位一致，已验证），在 Chikusei 128 波段（35 个 256² 训练 tile，21 个 256² 测试 tile）上比较空间扫描与光谱扫描。显存限制下训练 patch 用 128×128，`--mult 1,1,2`（15M）是推荐折中。详见 `docs/100bands.md`。V7 的 CAVE 结论为负之后，这条线是光谱扫描最后的验证场。

## 7. 待办（按优先级）

1. 主线转为「小波采样 + Mamba 空间扫描」（HyMCNetL3）：修正 HyMCNetL3 的 FFN 双重残差单独训练一次，量化影响（可能是廉价增益）；补 4 / 6 stage 版本与 SSR-M 的同参数对比。
2. 启动 Chikusei 线（`run_chik.sh`，先 V7NB 再 NB 对照），验证光谱扫描在 128 波段上是否翻盘。若有效再训 V7S 三档（σ=4 起，再扫 2 / 8）。
3. V7 救济（优先级低）：窗口 8→16 或空间 + 光谱并联扫描，针对中尺度空间上下文不足。
4. 论文指标用 MATLAB `cal_psnr_ssim.m` 复核 SSIM；测试集只有 10 景，0.1~0.2 dB 的差异要谨慎解读。
5. 若 Chikusei 上光谱扫描有效，用 ICVL 原始数据（519 波段）重采样到 128 波段复现（自然场景，与 CAVE 同类）。
