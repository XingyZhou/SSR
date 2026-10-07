# 任务文档:基于原生 Mamba 的光谱 SSM(固定 σ)

分支:`cursor/spectral-mamba-fixed-a-a97d`

## 1. 目标

在**原生 state-spaces/mamba** 的基础上,构造一个以学习**光谱连续性**为目的的 SSM 模块,
替换 `Model.py` 中沿波段方向扫描的光谱 token mixer。

核心理论:光谱与文本不同,**各波段没有重要程度差异**,相邻波段的关联只由光谱距离决定。
因此 Mamba 为文本设计的"时间轴选择性"(可学习 A、随输入变化的 dt)对光谱是有害的,
应替换为固定的物理先验;"内容轴选择性"(B、C)是否保留则作为消融对象。

## 2. 已完成的改动

| 文件 / 目录 | 状态 | 说明 |
|---|---|---|
| `mamba/` | 新增 | 原生 Mamba 源码,`state-spaces/mamba` 提交 `e9594ce`,已去掉 `.git` |
| `SpectralMambaNative.py` | 新增 | 本任务的主文件,见第 3 节 |
| `backup/` | 新增 | 根目录现有 `.py` / `.m` / `.md` 文件的原样副本,内容未改动 |
| `SpectralMamba.py` 等原文件 | 未动 | 之前自写的固定 A 版本仍可用,作为对照 |

## 3. `SpectralMambaNative.py` 设计

### 3.1 `FixedSigmaMamba(Mamba)` —— 继承原生 Mamba,只改三处

| 组件 | 原生 S6 | 本版本 | 原因 |
|---|---|---|---|
| A | 可学习 `A_log` 参数 | **固定 buffer**,`A[d,n] = -(n+1)/σ` | 衰减尺度 = 物理光谱宽度先验 |
| dt | `softplus(dt_proj(x_t))` 随输入变化 | **固定 = `dt_fixed`**(1 个波段间距):`dt_proj.weight = 0`,`bias = softplus⁻¹(dt_fixed)`,两者冻结 | 随输入变化的 dt 等于对不同波段施加不同注意力 |
| B, C | `x_proj(x_t)` 随输入变化 | 由 `bc_mode` 决定(见下) | 消融内容轴选择性 |

其余(in_proj、causal conv1d、SiLU 门控、D 跳连、out_proj、CUDA 内核)与原生完全一致。
波段间衰减在任意位置都是 `exp(-(n+1)·dt/σ)`,与内容无关。

### 3.2 `bc_mode` 三档

| `bc_mode` | B、C | 性质 |
|---|---|---|
| `'selective'`(默认) | 原生随输入变化 | **半选择 SSM**:距离核固定,读写按内容。既非 S4 也非 S6,是本工作可主张的中间态 |
| `'s4'` | 每通道一个可学习 `(d_inner, d_state)` 向量,与位置、内容无关 | **物理 S4**(LTI):学习如何组合 σ, σ/2, …, σ/N 这 N 个指数核 |
| `'const'` | B = C = 1 | 纯固定指数核滤波,下界对照 |

原生 CUDA 内核本身支持非变量的 `(d_inner, d_state)` B、C,所以 `'s4'` / `'const'`
直接传固定张量进 `mamba_inner_fn` / `selective_scan_fn`,无需改内核。
`'selective'` 直接调用父类 `forward`,代码路径与原生一致。

### 3.3 `SpectralMamba` 外壳

- 输入输出 `(B, C, H, W) → (B, C, H, W)`,接口与 `SpectralMamba.py` 一致。
- 每个 `p×p` 空间块作为一条长度为波段数 C 的序列,token 维度 `d_model = p·p`。
- 双向扫描:正向和反向各一个 `FixedSigmaMamba`,输出相加。
- 参数:`dim, patch=4, expand=2, d_state=8, d_conv=3, sigma=4.0, dt_fixed=1.0, bc_mode='selective', bidirectional=True`。

### 3.4 与 S4 / S6 的关系(论述口径)

- 全固定(`'const'` 或 `'s4'`)时 SSM 核心回到 LTI,**本质上接近 S4**,这一点要正视。
- 与 S4 的真实区别:A 不是学的而是物理先验;dt 是物理波段间距,可直接代入真实 Δλ,
  支持换传感器 / 非均匀采样;保留 Mamba 的 block 外壳和内核。
- 建议论述:"把时间轴的选择性替换为物理光谱先验,保留 / 消融内容轴的选择性",
  而不是"把 S6 改回 S4"。

## 4. 使用方式

```python
# Model.py 中替换 import 即可
from SpectralMambaNative import SpectralMamba
self.SpecMamba = PreNorm(dim, SpectralMamba(dim=dim, patch=4, sigma=4.0, bc_mode='s4'))
```

环境要求:CUDA GPU + 编译好的 `mamba_ssm`:

```shell
pip install ./mamba
```

## 5. 待办 / 注意

- [ ] **尚未运行**。当前开发环境无 PyTorch / GPU,`SpectralMambaNative.py` 只通过了语法检查。
      首次在 GPU 上使用前,先用小张量做前向,检查三档 `bc_mode` 的输出形状和梯度。
- [ ] 接入 `Model.py` 并跑三档消融:`selective` / `s4` / `const`,再与原生 S6(`mamba_ssm.Mamba`,A、dt 都可学习)对比。
- [ ] σ 作为超参数扫描(如 2 / 4 / 8 个波段)。
- [ ] `'selective'` 档在 `inference_params`(推理缓存)下走原生路径;`'s4'` / `'const'` 未支持推理缓存(训练 / 测试不需要)。
- [ ] 原生 Mamba 默认 `dt_min=0.001, dt_max=0.1` 对本版本无影响(dt 已固定),但若将来恢复可学习 dt,需按 28 波段的短序列调大。
