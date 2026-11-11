// Read-only ABI of the paged-KV block transfer. Submissions include this header
// and implement the single entry point at the bottom.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Transfer direction, named from the serving engine's point of view:
//   STORE:    paged engine buffers -> chunk objects
//   RETRIEVE: chunk objects -> paged engine buffers
#define LMC_DIR_STORE 0
#define LMC_DIR_RETRIEVE 1

// Engine paged-buffer layout family. hs is the per-head trailing width in
// elements; for the fused family it is the fused content size (2*head_size)
// and kv_size == 1.
//   SPLIT_HND: per-layer tensor [2, NB, NH, BS, HS]
//   FUSED_HND: per-layer tensor [NB, NH, BS, CS]
#define LMC_LAYOUT_SPLIT_HND 0
#define LMC_LAYOUT_FUSED_HND 1

// Chunk objects are contiguous token-major buffers of logical shape
// [kv_size, NL, chunk_tokens, NH*HS], on device or in mapped pinned
// host memory (chunk_side_is_host == 1; the pointers are then host virtual
// addresses valid on device under UVA).
typedef struct {
  // Device-resident array of NL entries; entry l is the device address of the
  // engine's layer-l paged tensor.
  const uint64_t* paged_layer_ptrs;
  // Chunk object base addresses; entries [0, num_objects) are valid.
  uint64_t chunk_ptrs[4];
  // Device-resident array of total_blocks int64 engine block indices.
  // Entry f maps chunk-side flat block f (object f / blocks_per_object,
  // block f % blocks_per_object within the object) to an engine block id.
  const int64_t* block_ids;

  int32_t num_objects;          // 1..4
  int32_t total_blocks;         // num_objects * blocks_per_object
  int32_t blocks_per_object;    // chunk_tokens / bs
  int32_t skip_prefix_n_blocks; // flat blocks [0, skip) are not transferred
  int32_t direction;            // LMC_DIR_*
  int32_t layout;               // LMC_LAYOUT_*
  int32_t kv_size;              // 2 for SPLIT_HND, 1 for FUSED_HND
  int32_t nl;                   // layers
  int32_t nb;                   // engine blocks in the paged buffer
  int32_t bs;                   // tokens per block (page size)
  int32_t nh;                   // kv heads
  int32_t hs;                   // per-head width in elements (fused: 2*head)
  int32_t element_size;         // bytes per element (2 = bf16)
  int32_t chunk_tokens;         // tokens per chunk object
  int32_t chunk_side_is_host;   // 1: chunk_ptrs are mapped pinned host
  int32_t _pad0;

  // Device scratch arena. Nothing may be assumed or left behind across calls.
  void* device_scratch;
  int64_t device_scratch_bytes;
} TransferParams;

#ifdef __cplusplus
extern "C" {
#endif

// The one entry point. All work must be enqueued on `stream`; no
// allocation, no private streams/events, no device-wide synchronisation.
void lmcache_paged_kv_block_transfer(const TransferParams* params,
                                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
