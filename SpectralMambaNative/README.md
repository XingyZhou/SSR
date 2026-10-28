# SpectralMambaNative

## 1. 核心理论

光谱与文本不同,**各波段没有重要程度差异**,相邻波段的关联只由光谱距离决定。
因此 Mamba 为文本设计的"时间轴选择性"(可学习 A、随输入变化的 dt)对光谱是有害的,
应替换为固定的物理先验;"内容轴选择性"(B、C)是否保留则作为消融对象。

## 2. 目录内容

| 文件 / 目录 | 说明 |
|---|---|
| `mamba/` | 原生 Mamba 源码,`state-spaces/mamba` **tag v2.2.2**(提交 `8ffd905`),已去掉 `.git`。见 §4 为何不是 `e9594ce` |
| `SpectralMambaNative.py` | 本任务的主文件,见 §3 |
| `test_spectral_mamba_native.py` | 模块级 GPU 测试,已在 RTX 4090 上全部通过(§5) |
| `Model_SpecMamba.py` | 根目录 `Model.py` 的可切换副本:`SRB` 中新增 `self.SpecMamba = PreNorm(dim, SpectralMamba(...))`,插在 CMB 与 SAB 之间;`opt.bc_mode='none'` 时与原版 `Net` 完全一致(基线)。与 `Model.py` 的差异仅限这几处(`diff --strip-trailing-cr` 可查) |
| `Train_SpecMamba.py` | `Train.py` 的副本,模型换成 `Model_SpecMamba.Net`,新增 `--bc_mode/--sigma/--dt_fixed/--patch/--d_state/--ckpt_dir/--epochs`;权重写到本目录 `Checkpoint_<bc_mode>_sigma<σ>/`,支持续训 |
| `smoke_net_SpecMamba.py` | 整网冒烟:4 档各跑一个真实尺寸 batch 的前向 + 反向(§5) |
| `backup/` | 任务开始时根目录 `.py` / `.m` / `.md` 文件的原样副本,内容未改动 |
| 根目录 `Model.py` / `Mamba_Model.py` 等 | **未动** |

## 3. `SpectralMambaNative.py` 设计

### 3.1 `FixedSigmaMamba(Mamba)` —— 继承原生 Mamba,只改三处

| 组件 | 原生 S6 | 本版本 | 原因 |
|---|---|---|---|
| A | 可学习 `A_log` 参数 | **固定 buffer**,`A[d,n] = -(n+1)/σ`(实现上仍叫 `A_log`,存 `log((n+1)/σ)`,父类 forward 不用改) | 衰减尺度 = 物理光谱宽度先验 |
| dt | `softplus(dt_proj(x_t))` 随输入变化 | **固定 = `dt_fixed`**(1 个波段间距):`dt_proj.weight = 0`,`bias = softplus⁻¹(dt_fixed)`,两者冻结 | 随输入变化的 dt 等于对不同波段施加不同注意力 |
| B, C | `x_proj(x_t)` 随输入变化 | 由 `bc_mode` 决定(见下) | 消融内容轴选择性 |

其余(in_proj、causal conv1d、SiLU 门控、D 跳连、out_proj、CUDA 内核)与原生完全一致。
波段间衰减在任意位置都是 `exp(-(n+1)·dt/σ)`,与内容无关(已用单位脉冲响应验证)。

### 3.2 `bc_mode` 三档

| `bc_mode` | B、C | 性质 |
|---|---|---|
| `'selective'`(默认) | 原生随输入变化 | **半选择 SSM**:距离核固定,读写按内容。既非 S4 也非 S6,是本工作可主张的中间态 |
| `'s4'` | 每通道一个可学习 `(d_inner, d_state)` 向量,与位置、内容无关。初始化 B = 1,C ~ N(0, 1/d_state) | **物理 S4**(LTI):学习如何组合 σ, σ/2, …, σ/N 这 N 个指数核 |
| `'const'` | B = C = 1(buffer) | 纯固定指数核滤波,下界对照 |

原生 CUDA 内核本身支持非变量的 `(d_inner, d_state)` B、C,所以 `'s4'` / `'const'`
直接传固定张量进 `mamba_inner_fn` / `selective_scan_fn`,无需改内核。
`'selective'` 直接调用父类 `forward`,代码路径与原生一致。
`'s4'` / `'const'` 下 `x_proj` 整个成为死路径(dt 列乘零权重,B/C 列不再使用),已冻结,参数统计里不计入。

### 3.3 `SpectralMamba` 外壳

- 输入输出 `(B, C, H, W) → (B, C, H, W)`;H、W 不整除 `patch` 时 reflect pad 后裁回。
- 每个 `p×p` 空间块作为一条长度为波段数 C 的序列,token 维度 `d_model = p·p`。
- 双向扫描:正向和反向各一个 `FixedSigmaMamba`,输出相加。
- 参数:`dim, patch=4, expand=2, d_state=8, d_conv=3, sigma=4.0, dt_fixed=1.0, bc_mode='selective', bidirectional=True`。

### 3.4 与 S4 / S6 的关系(论述口径)

- 全固定(`'const'` 或 `'s4'`)时 SSM 核心回到 LTI,**本质上接近 S4**,这一点要正视。
- 与 S4 的真实区别:A 不是学的而是物理先验;dt 是物理波段间距,可直接代入真实 Δλ,
  支持换传感器 / 非均匀采样;保留 Mamba 的 block 外壳和内核。
- 建议论述:"把时间轴的选择性替换为物理光谱先验,保留 / 消融内容轴的选择性",
  而不是"把 S6 改回 S4"。

## 4. 环境(已配好)

本机:torch 2.0.1+cu118、Python 3.11.7、nvcc 12.4、RTX 4090,包管理器 **uv**。

- 原计划的 `e9594ce`(2.3.2.post1)在本机**不可编译**:其 `setup.py` 要求 nvcc 与 torch 的 CUDA 主版本一致(12 vs 11.8),
  且依赖 `triton>=3.5`、`tilelang`,解析依赖会把 torch 换成 CUDA 13 版本。
  改用 **v2.2.2 官方预编译 wheel**(`cu118torch2.0cxx11abiFALSE-cp311`),本工作用到的
  `Mamba` / `mamba_inner_fn` / `selective_scan_fn` 接口在两版间完全一致。
- 已用 `uv pip install --no-deps` 安装:`mamba_ssm==2.2.2`、`causal_conv1d==1.4.0`;
  另装 `transformers==4.45.2`(仅 `mamba_ssm/__init__.py` 导入链需要),副作用:`huggingface-hub` 1.10.1 → 0.36.2。
- 重装命令(wheel 在 GitHub Releases):

```shell
uv pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
uv pip install "transformers<4.46"
```

## 5. 测试结果(`python test_spectral_mamba_native.py`,2026-10-07)

三档全部 PASS:
- 输出形状 `(2,28,32,32)`、输入/参数梯度有限;
- `A` 为 buffer 且 `A[0,:3] = [-0.25, -0.5, -0.75]`(σ=4);`dt_proj` 冻结、`softplus(bias) = 1.0`;
- `mamba_inner_fn` 快路径 == torch conv + `selective_scan_fn` 慢路径(误差 0);
- `'s4'` / `'const'`:CUDA 内核 == `selective_scan_ref` 纯 PyTorch 参考(误差 < 5e-7);
- `'const'` 单位脉冲响应 == `dt·Σₙ exp(-(n+1)·dt·k/σ)`(误差 1e-7),证明衰减核确实与内容无关。

参数量(`patch=4` → `d_model=16`,`d_inner=32`,`d_state=8`,双向):`selective` 4,480 / `s4` 4,416 / `const` 3,392(可训练)。

整网冒烟(`python SpectralMambaNative/smoke_net_SpecMamba.py`,`Net` stage=3,batch 2,256×256,随机输入,fwd+bwd 一步):

| `bc_mode` | 可训练参数 | 全部参数 | 峰值显存 | 梯度 |
|---|---|---|---|---|
| `none`(原版基线) | 1.8101 M | 1.8101 M | 6.42 GB | OK |
| `selective` | 1.8926 M | 1.8949 M | 10.18 GB | OK |
| `s4` | 1.8914 M | 1.9133 M | 9.80 GB | OK |
| `const` | 1.8730 M | 1.8949 M | 9.80 GB | OK |

SpecMamba 每阶段加 6 个(5 个 SSRB + 1 个 SRB),3 阶段共 18 个,新增参数约 0.08 M。显存增加主要来自 `(b·h/p·w/p)` 条序列的中间激活,batch 2 下 4090 充裕。

## 6. 使用方式

作为模块单独用:

```python
import sys; sys.path.insert(0, 'SpectralMambaNative')
from SpectralMambaNative import SpectralMamba
self.SpecMamba = PreNorm(dim, SpectralMamba(dim=dim, patch=4, sigma=4.0, bc_mode='s4'))
```

跑消融(在仓库根目录,原 `Train.py` / `Model.py` 不受影响):

```shell
python SpectralMambaNative/Train_SpecMamba.py --bc_mode none        # 原版 Net 基线
python SpectralMambaNative/Train_SpecMamba.py --bc_mode selective
python SpectralMambaNative/Train_SpecMamba.py --bc_mode s4
python SpectralMambaNative/Train_SpecMamba.py --bc_mode const
python SpectralMambaNative/Train_SpecMamba.py --bc_mode s4 --sigma 2   # σ 扫描
```

## 6.5 接入研究主线:V7S(2026-10-19)

主线(见`docs/weekly/周报_2026-10-19.md`)已放弃块内分频,转向 **V7 = HyMCNetL3 骨架 + Mamba 沿光谱维扫描**(`model_v7.py`,原生 S6,8×8 窗口 → `d_model=64`,序列 = 通道)。
V7 恰好是本模块的"原生对照",因此把 `FixedSigmaMamba` 直接接到 V7 上:

| 文件(根目录) | 说明 |
|---|---|
| `model_v7s.py` | `HyMCNetL3_V7S`:与 `model_v7.py` 唯一区别是 `SpecMamba_Att` 内的 `Mamba` 换成 `FixedSigmaMamba`(d_state=16, d_conv=4, expand=2, win=8 与 V7 完全一致)。`opt.bc_mode / sigma / dt_fixed` 控制 |
| `Train_v7s.py` | `Train_v7.py` + `--bc_mode {selective,s4,const} --sigma --dt_fixed --ckpt_dir`,权重默认 `./Checkpoint_v7s_<bc_mode>_sigma<σ>/` |
| `Test_v3.py` | 模块查找表加入 `model_v7s`:`--model HyMCNetL3_V7S --ckpt_dir ./Checkpoint_v7s_selective_sigma4` |

四臂消融:**V7(原生:A 可学、dt 随输入)** vs **V7S-selective** vs **V7S-s4** vs **V7S-const**。

整网冒烟(`.venv`,torch 2.8 / mamba_ssm 2.2.5,batch 2,256²,fwd+bwd 一步):

| 臂 | 可训练参数 | 全部参数 | 峰值显存 | 一层 A[0,:2] | dt |
|---|---|---|---|---|---|
| V7 原生 | 2.952 M | 2.952 M | 9.9 GB | [-1, -2](可学) | softplus(bias)≈0.026(随输入) |
| V7S selective | 2.912 M | 2.921 M | 9.8 GB | [-0.25, -0.5](固定) | 1.0(固定) |
| V7S s4 | 2.904 M | 2.983 M | 9.5 GB | 同上 | 同上 |
| V7S const | 2.843 M | 2.921 M | 9.5 GB | 同上 | 同上 |

三档梯度有限;光谱因果性成立(扰动第 14 个及以后通道,前 14 个通道输出变化为 0)。
值得注意的对比:原生 V7 初始化下每波段衰减 ≈ exp(-0.026·1) ≈ 0.97(近似不衰减、靠学习),V7S 为 exp(-1/σ)=0.78(σ=4),先验已内置。

**环境更正**:实际训练环境是仓库内 `.venv`(uv venv:torch 2.8.0+cu128、mamba_ssm 2.2.5、无 causal_conv1d、numpy 2.2),
不是 §4 的 anaconda。本模块在 `.venv` 下的 `test_spectral_mamba_native.py` 同样全部 PASS;无 causal_conv1d 时
原生 Mamba 与 `FixedSigmaMamba` 都走 torch conv1d + `selective_scan_fn` 慢路径,V7 / V7S 公平。

## 7. 待办

- [x] GPU 前向 / 反向 / 三档形状与梯度检查
- [x] 接入模型(`Model_SpecMamba.py`,可切换副本,原文件不动)并通过整网 fwd+bwd 冒烟
- [x] 接入主线 V7(`model_v7s.py` / `Train_v7s.py`),原生 S6 对照即 V7 本身
- [ ] 训练 V7S 三档(V7 原生已于 2026-10-19 12:05 在 tmux `train_v7` 启动,约 2 天);`Model_SpecMamba.py` 那条 SSR-Net 线保留但不再是主线
- [ ] 测试脚本:`Test.py` 的对应副本尚未做(`Net(opt)` 需带 `bc_mode` 等字段,加载对应 `Checkpoint_*` 即可)。
- [ ] 显存:`selective` 档 batch 2 峰值 10.2 GB;如要加大 batch 或 patch 更小,可考虑 `checkpoint_lvl` / 单向扫描。
- [ ] 当前根目录 `Train.py` 实际用的是 `Mamba_Model.HyMCNetL3`(沿展平空间序列扫描),与本副本所基于的 `Model.Net` 不是同一基线,比较结果时注意。
- [ ] σ 作为超参数扫描(如 2 / 4 / 8 个波段)。
- [ ] `'selective'` 档在 `inference_params`(推理缓存)下走原生路径;`'s4'` / `'const'` 未支持推理缓存(训练 / 测试不需要,调用会抛 `NotImplementedError`)。
- [ ] 原生 Mamba 默认 `dt_min=0.001, dt_max=0.1` 对本版本无影响(dt 已固定),但若将来恢复可学习 dt,需按 28 波段的短序列调大。
