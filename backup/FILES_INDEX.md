# 备份文件索引

| 文件 | 作用 |
|---|---|
| `Cal_para_flops.py` | 统计模型参数量与 FLOPs (fvcore) |
| `Dataset.py` | CAVE/测试集 .mat 数据读取与 Dataset 封装 |
| `Exp_continuous_spectrum.py` | 实验: 连续光谱 Mamba, 零样本传感器/波段数适配 |
| `Exp_kernel_interpret.py` | 实验: 光谱混合器核可解释性 (冲激响应 vs 冻结 sigma) |
| `Exp_spectral_continuity.py` | 实验: 冻结 sigma 的光谱连续性 Mamba 对比 |
| `Model.py` | SSR 主网络(含 SpecMamba 光谱 token mixer) |
| `SpectralMamba.py` | 固定 A=-(n+1)/sigma 的自定义光谱 Mamba (SpectralMamba, FixedASSM, selective_scan_ref) |
| `Test.py` | 测试脚本: 用训练得到的权重重建并保存结果 |
| `Test_pretrain.py` | 测试脚本: 使用预训练权重 Checkpoint_pretrain |
| `Train.py` | 训练脚本 (CAVE 数据集) |
| `Utils.py` | 通用工具: 数据/掩码/指标/日志等 |
| `test_rgb_hsi.py` | 概念验证: 分辨率无关的 RGB->HSI (连续 Mamba) |
| `cal_psnr_ssim.m` | Matlab: 计算 PSNR/SSIM |
| `cal_ssim.m` | Matlab: SSIM 计算函数 |
| `README.md` | 项目说明 |
| `IDEAS_ARCHIVE.md` | 备选研究方向归档 |
| `LICENSE` | 许可证 |

新增: `SpectralMambaNative.py` (基于原生 mamba_ssm.Mamba 的重新实现), `mamba/` (原生 Mamba 源码)。
