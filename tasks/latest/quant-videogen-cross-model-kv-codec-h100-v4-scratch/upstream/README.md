<div align="center">

# [ICML 2026] QuantVideoGen

**Quantized KV-cache compression for video generation models**

<p>
  <a href="https://svg-project.github.io/qvg/"><img src="https://img.shields.io/badge/Website-76B900?style=for-the-badge&logo=safari&labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2602.02958"><img src="https://img.shields.io/badge/Arxiv-B31B1B?style=for-the-badge&logo=arxiv&labelColor=555555"></a>
  <a href="#"><img src="https://img.shields.io/badge/ICML-2026%20Accepted-1f77b4?style=for-the-badge&labelColor=555555"></a>
  <a href="#"><img src="https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&labelColor=555555"></a>
</p>

</div>

> **~7× smaller KV cache, ~85% less memory, no fine-tuning, no weight changes.** QuantVideoGen quantizes the KV cache of autoregressive video generators to **INT2** during inference and decompresses it on the fly at attention time — turning the cache from "the dominant memory cost of long video generation" into "an afterthought."

<p align="center">
  <img src="assets/qvg_teaser.png" width="95%" />
</p>

QuantVideoGen is a lightweight KV-cache quantization toolkit for autoregressive video generation. It compresses long-horizon attention cache during inference, with experiment integrations for LongCat-Video, Self-Forcing, HY-WorldPlay, and LingBot-World.

## ✨ Highlights

- Quantizes KV cache with Triton k-means / staged product quantization kernels — **~7× compression** at INT2.
- Keeps the original model weights unchanged; quantization is applied to inference cache.
- Includes bf16 and quantized launch scripts for four long-video / streaming generation repos.
- Targets memory-heavy long-context settings where KV cache dominates peak usage.

<a id="installation"></a>

## 📦 Installation

```bash
conda create -n qvg python=3.12.9 -y
conda activate qvg

pip install uv

# Everything, recommended for reproducing all experiments.
uv pip install -e ".[all]"

# Or install only one experiment extra.
uv pip install -e ".[longcat]"
uv pip install -e ".[selfforcing]"
uv pip install -e ".[hyworldplay]"
uv pip install -e ".[lingbot]"

# Flash Attention, CUDA 12 / torch 2.8 wheel.
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

<a id="download-models"></a>

## ⬇️ Download Models

Download model checkpoints before running experiments.

```bash
# LongCat-Video
hf download meituan-longcat/LongCat-Video --local-dir ckpts/LongCat-Video

# Self-Forcing
bash scripts/Self-Forcing/download_models.sh

# HY-WorldPlay
bash scripts/HY-WorldPlay/download_models.sh

# LingBot-World
bash scripts/LingBot/download_models.sh
```

<a id="quick-start"></a>

## 💥 Quick Start

Each integration provides a bf16 baseline script and a quantized script. The quantized scripts currently use `triton-nstages-kmeans-int2` with block size 64 and 256 K/V centroids by default.

```bash
# LongCat-Video
bash scripts/LongCat/run_bf16.sh
bash scripts/LongCat/run_qvg.sh

# Self-Forcing
bash scripts/Self-Forcing/run_bf16.sh
bash scripts/Self-Forcing/run_qvg.sh

# HY-WorldPlay
bash scripts/HY-WorldPlay/run_bf16.sh
bash scripts/HY-WorldPlay/run_qvg.sh

# LingBot-World
bash scripts/LingBot/run_bf16.sh
bash scripts/LingBot/run_qvg.sh
```

Outputs are written under `results/`. Quantization options can be changed directly in the corresponding `run_qvg.sh` script.

<a id="memory-results"></a>

## 📊 Memory Results

The table below reports KV-cache memory for the provided scripts. Numbers are in MB.

| Model | Precision | QVG | Per Layer KV | Total KV Cache | Compression Rate |
| --- | ---: | :---: | ---: | ---: | ---: |
| LongCat-Video  | BF16 | ✗ | 464.00   | 22272.00 | 1.00x |
| LongCat-Video  | INT2 | ✓ |  67.32   |  3231.28 | **6.89x** |
| Self-Forcing   | BF16 | ✗ | 1535.76  | 46072.88 | 1.00x |
| Self-Forcing   | INT2 | ✓ |  220.45  |  6613.59 | **6.97x** |
| HY-WorldPlay   | BF16 | ✗ |  990.00  | 29700.00 | 1.00x |
| HY-WorldPlay   | INT2 | ✓ |  141.18  |  4235.45 | **7.01x** |
| LingBot-World  | BF16 | ✗ |  884.00  | 35360.04 | 1.00x |
| LingBot-World  | INT2 | ✓ |  142.33  |  5693.36 | **6.21x** |

Across these runs, QuantVideoGen consistently delivers **~6–7× KV-cache compression** and reduces total KV-cache memory by **about 85%**.

<a id="citation"></a>

## ✏️ Citation

```bibtex
@article{xi2026quant,
  title={Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization},
  author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Li, Muyang and Cai, Han and Li, Xingyang and Lin, Yujun and Zhang, Zhuoyang and Zhang, Jintao and Li, Xiuyu and others},
  journal={arXiv preprint arXiv:2602.02958},
  year={2026}
}
```
