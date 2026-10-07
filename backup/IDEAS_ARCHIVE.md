<!-- [备份注释] 文件名: IDEAS_ARCHIVE.md | 作用: 备选研究方向归档 -->

# Alternative Directions for Spectral Mamba in HSI

This document archives two alternative research directions for integrating Mamba into Hyperspectral Image (HSI) Reconstruction. We are currently pursuing **Direction 2 (Continuous-Spectrum Mamba for Zero-Shot Sensor Adaptation)**, but these remain viable fallbacks or future work.

## Direction 1: Spatially-Adaptive Physical Spectral Kernel

**Core Idea:**
Instead of a single global spectral width $\sigma$ or a freely learned, uninterpretable state transition matrix $A$, we enforce a strict physical kernel where the *only* degree of freedom is a spatially varying spectral width $\sigma(x, y)$.

**How it works:**
1. Strip all input-dependent gates, $B$, $C$, and $D$ projections from the Mamba block.
2. The SSM step is strictly: $h_t = \exp(-\Delta / \sigma(x, y)) h_{t-1} + x_t$.
3. A lightweight spatial CNN predicts $\sigma(x, y)$ from the spatial features.
4. Because the network cannot use $B/C$ projections to bypass the exponential decay, the optimizer is *forced* to learn a meaningful width map. Flat regions will learn a large $\sigma$ (strong smoothing), while edges/absorptions will learn a small $\sigma$ (preserving sharp peaks).

**Selling Point:**
An interpretable, spatially-adaptive physical prior embedded directly into the Deep Unfolding optimization, visualized through meaningful width maps.

---

## Direction 3: Unmixing the Spectral Dispersion (CASSI-Native Scan)

**Core Idea:**
Current approaches (including our initial ones) apply spectral mixers on the 3D datacube *after* the initial data consistency step has roughly aligned the bands. However, the physical reality of CASSI is that spectral bands are shifted spatially.

**How it works:**
1. Instead of scanning along the $C$ dimension of a $H \times W \times C$ cube, the Mamba scans along the *dispersion trajectory* in the 2D measurement space (or the shifted 3D space).
2. The scan explicitly models the overlapping of different wavelengths onto the same physical sensor pixels.
3. The $\sigma$ in this context relates to the Point Spread Function (PSF) and the dispersion blur of the CASSI system, rather than just material smoothness.

**Selling Point:**
The first continuous state-space model that natively integrates with the CASSI physical dispersion process, doing unmixing via recurrent state propagation rather than static CNN receptive fields.
