# Clustered-PCA residual KV codec with interleaved rANS

Per tensor `[B,H,S,D]`, each (batch, head) row is coded independently:

1. **Clusters** — deterministic k-means (strided init, 10 Lloyd iterations, up to
   256 centroids) on the joint, per-row normalized [1.3 K, V] features of
   the tokens; keys and values share the cluster ids (one aliased, packed tensor
   referenced by both payloads) and each keeps its own centroids. Fused Triton kernels do the distance
   + argmin and the one-hot-GEMM cluster sums (no float atomics, so encoding is
   bit-reproducible). Centroids are stored as int8 with one scale per
   (row, dim) plus the FP32 row mean; ids are uint8.
2. **Residual transform** — PCA basis of the residual covariance (batched
   `gesvda`, `eigh` fallback). The basis is stored as int8; coefficients use the
   quantized basis' exact inverse, so basis rounding adds no reconstruction error.
3. **Quantization** — trellis-coded quantization (8-state TCQ, union
   codebooks of even/odd multiples of the TCQ step) along each token's PCA
   coefficients, found by a Triton Viterbi kernel. The search minimizes squared
   error only: a rate term (ECTCQ) saves bits but shrinks values toward zero,
   and shrinkage of decoded keys lowers their logit variance and systematically
   pulls softmax mass away from compressed history. TCQ step =
   `0.5 x step x RMS(coefficients)`; the error is shaped by scaling component
   j's step by `(lam_j/mean lam)^-gamma` (folded into the stored basis rows).
   Keys: step 2.0, gamma 0.2 (finer on high-variance directions, which
   dominate query-key logits). Values: step 2.1, gamma -0.1 (relatively finer
   on low-variance directions, which matter for long-horizon quality). Each
   head's step is further scaled by `(rms_head / rms)^alpha` (keys -0.25,
   values -0.75), i.e. finer steps for high-energy heads. A quality floor shrinks the step until the
   relative L2 error is <= 0.6 (only matters for unstructured inputs).
4. **Entropy coding** — rANS (32-bit state, 16-bit words, 12-bit probabilities)
   over codebook indices {0, +-1, ..., +-4, ESC} with one frequency table per
   (row, component, codebook parity), storing only the 9 interior cumulative
   boundaries; the decoder tracks each lane's trellis.
   The 32 lanes of a warp share one interleaved word stream; the decoder uses
   warp ballot/popc/shuffle (inline PTX) so a step has no reductions or
   branches. Values with |q| > 4 are stored raw (int8) and located by ballot.
5. **Side information** — losslessly packed: int8 centroids and basis use b-bit
   bit-plane packing (b chosen per tensor) with raw escapes; cluster ids store a
   "same as previous token" flag plus the ids of non-repeating tokens.
6. **Decode** — the rANS kernel writes int8 coefficients; a Triton GEMM kernel
   rebuilds `mean + centroid + step * (q @ basis)` straight to BF16.
   `decode(p, p)` decodes the shared payload once.

Operating point: chosen with an attention-output error proxy measured on late
chunks (all history compressed), calibrated against evaluator accept/reject
results.
