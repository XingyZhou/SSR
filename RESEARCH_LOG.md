# Spectral Mamba 研究记录

分支:`cursor/spectral-mamba-fixed-a-a97d`。时间跨度 2026-10-01 至 2026-10-07。
备选方向见 `IDEAS_ARCHIVE.md`。所有实验输出在 `exp_out/`。

## 1. 主线想法

沿波段轴做双向 S6 扫描,状态矩阵 A 固定为 `-(n+1)`,有效衰减为 `-(n+1)/σ`。σ 是光谱相关宽度(波段数或 nm),目的是让 SSM 的核带有物理含义,而不是自由学习 A。

关键代码:
- `SpectralMamba.py`:`FixedASSM`、`SpectralMamba`。支持 `width_mode` = fixed/param/estimate,`selective_dt`,`fixed_A`(可学习 A 消融),以及传入物理 `wavelengths` 作为步长。
- `Model.py`:`--spectral_mamba` 时在 `dim==bands` 的 SRB 里用 SpecMamba 替换 CMB+SAB(含 SSRU 的 down1/up1),默认关闭,旧 checkpoint 可直接加载。
- CLI:`--spectral_mamba --sm_width_mode --sm_patch --sm_d_state --sm_fixed_dt`。

## 2. 时间线与结论

| 日期 | commit | 内容 | 结论 |
|---|---|---|---|
| 10-01 | 9d7ee2b | 加入 SpecMamba 分支 | - |
| 10-01 | d6f2511 | `Exp_spectral_continuity.py`:合成 GP 光谱 + Indian Pines 丢波段恢复 | - |
| 10-01 | ecb94bc | 修复不学习:`in_proj` 加 bias,改用默认 Linear 初始化 | 无 bias 时全零波段 token 经 SiLU 门控被静音;std=0.02 初始化使输出成为三个近零因子之积,落在鞍点 |
| 10-01 | 3de68d9 | `selective_dt=False` | 选择性 dt 会自行调衰减,σ 估计器停在初始值;固定 dt 才能让 σ 成为唯一控制量 |
| 10-01 | 2a5c49a | σ 冻结为超参数,dt 固定为一个波段 | - |
| 10-01 | 52f3a79 | 冻结 σ 的丢波段实验 | 30% 丢波段下,真实数据 19 dB(输入 5.7 dB;无光谱信息的空间 CNN 13 dB,光谱 MLP 22 dB,所以先验可用但不是最优);合成数据 17 dB,但预测均值的平凡基线约 16.5 dB(空间 CNN 为 16.5 dB),不能说明问题;块的 Jacobian 半宽度在结果文件里为 0.0,可能是残差恒等项主导所致,未验证;σ 匹配 GP 长度尺度只在 ℓ=2 时 +0.3 dB,ℓ=8 变差,σ 小一点略好 |
| 10-01 | 6d261a9 | 2 阶段小 patch 训练可在 CPU/GPU 跑 | 无 CAVE 时回退 Indian Pines;每 epoch 记录 PSNR/SAM |
| 10-01 | eb2c01b / d25e810 | 用 SpecMamba 替换 CMB+SAB | 64×64、8 epoch、CPU:baseline 26.33 dB / 5.52°,SpecMamba 25.05 dB / 6.53°,约慢 3 倍(13 s vs 4 s/epoch)。仅 8 epoch,非收敛结果 |
| 10-04 | 5bba8c7 / f25eb40 | `Exp_kernel_interpret.py` 核可解释性 | 裸 SSM(B=C=1, Δ=1)的 e-folding 与平均滞后随 σ 增大;放进完整 SpecMamba/SRB(学习的 B/C、门控、skip)后,非对角 Jacobian 宽度对所有 σ 都约 8 个波段,与可学习 A 和 CMB 无差别。**σ 的物理含义被 B/C/门控绕开** |
| 10-06 | 4390b7e | 物理波长作步长;`Exp_continuous_spectrum.py` 零样本传感器迁移 | 见下表,物理 dt 泛化不如 learned dt;提交说明的解释是 B/C 过拟合源波段 |
| 10-06 | 88418b1 | `test_rgb_hsi.py` RGB→HSI 概念验证 | 合成数据、16×16,只有图 `exp_out/rgb2hsi/zero_shot_rgb2hsi.png`,无定量指标 |
| 10-06 | e02c785 | `IDEAS_ARCHIVE.md` 存档方向 1、3 | - |

### 零样本传感器迁移(PSNR, dB,`exp_out/continuous/results.json`)

| 设定 | source 28 | target 15 均匀 | target 20 带间隙 |
|---|---|---|---|
| learned dt | 38.00 | 35.73 | 36.33 |
| continuous(物理 dt) | 37.59 | 33.65 | 33.07 |

## 3. 现状判断

1. σ 的物理意义没有保住,原因是可学习的 B/C/门控旁路了衰减(10-04、10-06 两次实验一致)。
2. 目前没有性能优势:重建对比低 1.3 dB,零样本迁移低 2 到 3 dB。
3. 实验规模偏小:64×64 patch、CPU、8 到 12 epoch,部分用 Indian Pines 代替 CAVE,不能作为论文结论。

## 4. 下一步:迁移到 RGB→HSI(NTIRE2022 ARAD-1K)

**状态:数据集尚未获取。** 云环境网络策略拒绝了 `huggingface.co`、`zenodo.org`、`github.com`(403)。需要在环境设置里放开 Network access,或手动提供数据。环境里也未安装 torch,训练需要 GPU。

### 模型设计
- 空间编码器(轻量 CNN 或 MST++ 式注意力)把 RGB 变成特征 F(x,y)。
- 序列构造:`token_λ = F(x,y) + e(λ)`,e 为波长嵌入。
- 光谱混合器:`FixedASSM`,衰减 `exp(-(n+1)·Δλ/σ)`,Δλ、σ 均用 nm。
- 输出头线性映射到反射率。
- 可选物理约束:`‖CRF·Ĥ − RGB‖` 重投影损失(ARAD-1K 提供 CRF)。

### 消融(针对"B/C 旁路 σ")

| 变体 | B/C | 门控与 skip | 目的 |
|---|---|---|---|
| A. Full | 输入相关 | 有 | 上界 |
| B. Const-BC | 可学习常量 | 有 | 检验 B/C 是否为主因 |
| C. Strip | B=C=1 | 无 | IDEAS_ARCHIVE 方向 1 的纯物理核 |
| D. σ(x,y) | 同 C | 无 | 方向 1 完整版,CNN 预测空间 σ |

每个变体用 `Exp_kernel_interpret.py` 的方法检查 Jacobian 宽度是否随 σ 变化。

### 实验计划
1. ARAD-1K 官方划分上对比 MST++ / HSCNN+,指标 MRAE、RMSE、PSNR、SAM。
2. 上述 A 到 D 的消融,σ 取多个固定值与空间预测。
3. 零样本波段数迁移:训练 31 波段(10 nm),测试 16/61/121 波段与带间隙。真值只有 31 波段,上采样只能与插值参考比较。
4. 对照:同一空间编码器配 1×1 卷积输出 31 通道,验证连续光谱版本在固定 31 波段上不退化。

### 风险
- 物理 dt 的零样本迁移已经失败过一次,若 RGB→HSI 仍失败,需要重新考虑这条路。
- σ 的取值需要先在 ARAD-1K 上统计真实光谱自相关长度来校准,PoC 里的 20 nm 未校准。
- 每像素 31 步扫描较慢,训练用 128×128 patch,并启用 `mamba_ssm` CUDA kernel(接口已有,需先把 σ 缩放换成 nm 单位)。

### 执行顺序
1. 数据加载与评估脚本。
2. `RGB2HSI.py`,先跑变体 A。
3. 变体 B、C、D 与消融。
4. 零样本分辨率实验。
