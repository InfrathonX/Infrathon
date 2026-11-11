# Paper Algorithm 1: standard PyTorch K-Means

`standard_kmeans.py` is a direct PyTorch implementation of the paper's
standard one-iteration pseudocode:

| Paper stage | PyTorch operation |
| --- | --- |
| Full distance matrix | FP32 `torch.bmm` plus the two squared norms |
| Argmin | `Tensor.argmin(dim=-1)` |
| Atomic sums | FP32 `Tensor.index_add_` |
| Atomic counts | int32 `Tensor.index_add_` |
| Normalization | elementwise division |

The implementation deliberately materializes the complete `[B,N,K]` distance
matrix.  There is no point or centroid chunking, no streaming minimum, and no
fused assignment kernel.  Consequently its distance storage alone is
`4*B*N*K` bytes.

Algorithm 1 does not define empty-cluster behavior.  The task adapter retains
the old centroid for empty clusters so that the implementation has the v7-scratch
`assign`/`update` ABI.  This is the only semantic bridge beyond batching and
returning integer counts.

Build the deterministic ordinary-candidate archive with:

```bash
python3 host_reference/standard_pytorch_kmeans/build_archive.py \
  --output artifacts/standard-pytorch-kmeans-baseline.tar.gz \
  --source-output \
    host_reference/standard_pytorch_kmeans/generated_task_adapter.py
```

Then evaluate it through the unchanged Judge:

```bash
JUDGE_GPU=3 ./scripts/eval_archive.sh \
  artifacts/standard-pytorch-kmeans-baseline.tar.gz
```
