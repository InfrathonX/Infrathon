# Quant-VideoGen KV codec: Claude Opus 5.5 vs. the human codec

This directory holds two codecs for the [Quant-VideoGen KV Codec](../../tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch) task. Both use the same task interface, `encode(k, v)` / `decode(packed_k, packed_v)`, so either one can be dropped into `submission/` and evaluated the same way.

| Directory | Codec |
| --- | --- |
| [`claude/`](claude) | The final codec Claude Opus 5.5 wrote in one 24-hour run, starting from an empty `submission/`. It is the run's 22nd and last submission, unchanged. `notes.md` is the agent's own design note. |
| [`human-qvg/`](human-qvg) | The official [Quant-VideoGen](https://github.com/svg-project/Quant-VideoGen/tree/0601468f2dbba6a17ac7086faec6d41527cad188) codec. `codec/` is upstream `quant_videogen/` at commit `0601468f`, copied byte for byte; only the package directory is renamed. `mlsysforge_codec.py` wraps it with the official `run_qvg.sh` settings. It is the task's reference codec ([`oracle/`](../../tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/oracle)). |

## Results

| Video model / dataset (200 samples each) | Compression ↑ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- |
| HY-WorldPlay / WorldScore | 6.75× → **20.89×** | 31.15 → **36.33** | 0.923 → **0.975** | 0.059 → **0.023** |
| LongCat-Video / MovieGen | 6.88× → **18.83×** | 23.67 → **26.95** | 0.836 → **0.884** | 0.126 → **0.085** |

Entries are `human-qvg` → `claude`. PSNR, SSIM and LPIPS compare the first segment generated with compressed KV against uncompressed BF16. None of these samples are in the task's test cases. See the [paper](https://arxiv.org/abs/2611.11995) (Section 5 and the appendix) for the setup.

Averaged over the two datasets, compression rises from **6.82× to 19.86×**. On the task's own 17 test cases, the Judge scored the same codec at **19.62×**, the geometric mean of 18.76× on LongCat-Video and 20.53× on HY-WorldPlay. That is the number in the main README's run figure.

## How the two codecs differ

Both cluster each attention head's tokens and store a centroid id plus a residual per token. They differ in how the residual is stored.

| | `human-qvg` | `claude` |
| --- | --- | --- |
| K/V clustering | Separate for K and V, 256 centroids | Shared ids from joint `[1.3·K, V]` features, `min(256, S/128)` centroids |
| Residual transform | None | PCA, all dimensions kept |
| Quantization | Each value to {−s, 0, s}, one FP8 scale per 64 values | 8-state trellis-coded quantization, finer steps on high-variance key directions and high-energy heads |
| Storage | 2 bits per value | rANS entropy coding, plus lossless packing of centroids, bases and ids |

Neither codec checks which model it is serving. All of `claude/`'s settings are fixed constants or rules computed from the input tensor. `human-qvg/` uses the head count only to pick its k-means iteration count (100 for LongCat-Video, which has 32 heads; 2 otherwise), to match the official per-model scripts.

## Evaluate a codec

First follow steps 1–2 of the [task README](../../tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/README.md) to fetch the assets and build the images. Then package a codec as a submission and run the Judge on six GPUs:

```bash
CODEC=claude            # or human-qvg
mkdir -p /tmp/$CODEC/submission
cp -r examples/quant-videogen-kv-codec/$CODEC/. /tmp/$CODEC/submission/
tar czf /tmp/$CODEC.tar.gz -C /tmp/$CODEC submission
JUDGE_GPUS=1,2,3,4,5,6 tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/scripts/eval_archive.sh /tmp/$CODEC.tar.gz
```

## Limitations

- **Resident memory, not compute.** `claude/` shrinks the GPU memory that holds the persistent KV cache. KV is still decoded back to BF16 before attention, as in `human-qvg/`.
- **Decoding cost.** rANS decoding makes each `claude/` decode call more expensive than a `human-qvg/` one. On LongCat-Video, which decodes each cached segment about 56 times, end-to-end generation is about 2% slower. On HY-WorldPlay it is faster, mostly because `human-qvg/` re-tunes its Triton kernels whenever the context length changes.
- **One operating point.** The comparison covers two video models and Quant-VideoGen's INT2 setting.

## License

`claude/` is covered by this repository's [license](../../LICENSE). `human-qvg/codec/` is third-party code by the Quant-VideoGen authors; see [svg-project/Quant-VideoGen](https://github.com/svg-project/Quant-VideoGen) for its terms.
