# Official adapter, pre-Hopper compatibility variant

`upstream.patch` is the minimal diff against the official checkout
`12ed49b2eddc207dfc5ed27dba244af2365bd9c6` (`flash_kmeans/assign_euclid_triton.py`)
that lets the official adapter launch on GPUs the upstream heuristics do not
know. It is applied to a scratch copy of the upstream checkout by
`scripts/make_ampere_compat_oracle_archive.sh`; nothing here is a Docker build
input and `official_oracle/` stays byte-exact.

What the patch changes, and why each change is a no-op on H100/H200/A100:

1. `_KNOWN_ARCHS` / the small-D and large-D pickers learn `A6000` and `A40`
   and route them to the A100 (same Ampere generation) tables. Unknown GPU
   names made upstream force *every* shape through the split-D kernel.
2. `_fit_config_to_smem_split_d` keeps `BLOCK_D` a power of two
   (`max(_next_pow2(D), 16)` instead of `max(D, 16)`); `tl.arange` rejects
   `BLOCK_D=31`, which the forced split-D path produced for `D=31`.
3. The fp16/bf16 tables are passed through `_fit_config_to_smem` instead of
   being returned unfitted. Fitting returns the table entry unchanged whenever
   it already fits, so parts with >= 164 KiB of opt-in shared memory keep
   byte-for-byte behaviour; only smaller-SMEM parts get shrunk tiles.

Together with the builder flag `--device-smem-budget` (which replaces the fixed
`232448`-byte H100 budget by
`min(props.shared_memory_per_block_optin, 232448)`, still without dynamic
`getattr`), the generated archive is
`artifacts/official-oracle-ampere-compat.tar.gz`. It is a different candidate
identity from `official-oracle.tar.gz` and must be reported as such; it is not
a replacement for the H100-accepted archive.
