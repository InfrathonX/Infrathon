// Paper/upstream reference ("full LMCache system path") for the LMCache
// paged-KV block transfer task.
//
// Upstream: LMCache/LMCache @ ebe3edde1df6813a14a5d69ae36a5109572f0b10
// (2026-10-29, latest main at the time of this port).  Vendored pieces:
//   * csrc/cuda/mp_mem_kernels_detail.cuh   -- device code of
//     multi_layer_block_transfer_kernel (verbatim for the two HND formats
//     the task scores: NL_X_TWO_NB_NH_BS_HS == LMC_LAYOUT_SPLIT_HND and
//     NL_X_NB_NH_BS_CS == LMC_LAYOUT_FUSED_HND).  Unchanged vs the task's
//     pinned commit f9addd2e for these formats.
//   * csrc/cuda/mp_mem_kernels.cu           -- host launch geometry of
//     multi_layer_block_kv_transfer_templated (verbatim arithmetic) and the
//     staged executor execute_object_group_transfer (H2D: staging copies
//     then kernel; D2H: kernel then staging copies).
//   * csrc/cuda/mem_kernels.cu              -- lmcache_memcpy_async (one
//     cudaMemcpyAsync per pinned region; each Judge chunk object is its own
//     pinned allocation, so one copy per object).
//   * lmcache/v1/multiprocess/{modules/lmcache_driven_transfer.py,
//     object_group_transfer.py} -- _run_object_group_transfer_plan: host
//     memory objects are never accessed zero-copy by the kernel; they are
//     staged whole into GPU temp buffers ("temp object-group buffers", one
//     per batch slot) with cudaMemcpyAsync, batched, and the kernel runs on
//     the GPU temp buffers.  The direct copy-engine path
//     (cudaMemcpyBatchAsync, upstream #5308/#5506) is ineligible for HND
//     layouts upstream (resolve_block_addressing returns false for HND), so
//     HND always takes this staged path upstream.
//
// ABI / glue adaptations (the only non-upstream logic):
//   1. GPU temp buffers = the Judge's device scratch arena; batch size =
//      number of whole objects that fit in it (upstream: max_batch_size temp
//      buffers of one object each).
//   2. Device-resident chunk objects (chunk_side_is_host == 0): the kernel
//      runs directly on them (no staging needed).
//   3. STORE with a skipped prefix: upstream copies whole temp objects back
//      to the host object; the task contract requires the skipped prefix of
//      the host chunk to stay untouched, so the D2H staging copy is restricted
//      to the non-skipped token range of each (kv, layer) slab
//      (cudaMemcpy2DAsync), and fully skipped objects are not copied.
//      Unskipped objects use the upstream whole-object cudaMemcpyAsync.
//   4. TORCH_CHECK -> early return; current-stream lookup -> ABI stream.
//   5. If an object does not fit the arena (never for the scored grid), fall
//      back to the zero-copy kernel (the task seed's behaviour).
// SPDX-License-Identifier: Apache-2.0 (derived from upstream LMCache).

#include <type_traits>
#include <stdint.h>
#include <stddef.h>

#include "transfer_abi.h"

namespace {

#define PORT_CHECK(cond) \
  do {                   \
    if (!(cond)) {       \
      return;            \
    }                    \
  } while (0)

// --- csrc/kv_transfer_plan_types.h: PageBufferShapeDesc (block_stride_elems
// is 0 == tight for the HND formats) ---------------------------------------
struct PageBufferShapeDesc {
  int kv_size;
  int nl;
  int nb;
  int bs;
  int nh;
  int hs;
  int element_size;
  int block_stride_elems;

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }
  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }
  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

// --- csrc/cuda/mp_mem_kernels_detail.cuh (HND formats only) ----------------
template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;
};

template <typename ScalarType, int format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == LMC_LAYOUT_SPLIT_HND) {
    // Normal HND: L tensors [2, NB, NH, BS, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else {
    // Content-size HND: L tensors [NB, NH, BS, CS]
    return engine_block_idx * scalars_per_block;
  }
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  // HND: [NH, BS, HS] -- heads are outermost within a block
  size_t scalars_per_head_block = shape_desc.bs * scalars_per_head;
  return head_idx * scalars_per_head_block + token_offset * scalars_per_head;
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v, const int token_offset_in_lmcache_object,
    const int layer_idx, const int lmcache_chunk_size,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // LMCache is using 2LTD all the times
  return token_offset_in_lmcache_object * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
}

__device__ inline void st_cs(uint4* addr, uint4 val) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

template <typename ScalarType>
__device__ inline void warp_copy(ScalarType* __restrict__ dst,
                                 const ScalarType* __restrict__ src,
                                 size_t num_elements) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (size_t i = idx; i < num_elements; i += stride) {
      st_cs(dst + i, ld_cs(src + i));
    }
  } else {
    for (size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, int format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size) {
  const int head_idx = threadIdx.y;
  const int init_token_offset = threadIdx.z;
  const int token_stride = blockDim.z;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t lmcache_global_offset =
      calculate_lmcache_global_offset<ScalarType, format>(
          k_or_v, offset_in_lmcache_block, layer_idx, lmcache_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr =
      (ScalarType*)paged_buffer_ptrs[layer_idx];

  for (int token_offset = init_token_offset; token_offset < shape_desc.bs;
       token_offset += token_stride) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t lmcache_local_offset =
        calculate_lmcache_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      warp_copy<ScalarType>(engine_ptr, lmcache_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(lmcache_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, int format>
__global__ void multi_layer_block_transfer_kernel(
    MemoryObj4<ScalarType> lmcache_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids, const int num_blocks_per_object,
    const PageBufferShapeDesc shape_desc, const int lmcache_chunk_size,
    const int skip_prefix_n_blocks) {
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;

  const int engine_block_idx = engine_block_ids[flat_block_idx];
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs, shape_desc, lmcache_chunk_size);
}

// --- csrc/cuda/mp_mem_kernels.cu: multi_layer_block_kv_transfer[_templated]
// (one launch over num_objects object pointers, a block_ids slice, a skip) --
struct LaunchArgs {
  const uint64_t* paged_layer_ptrs;
  uint64_t objects[4];
  int num_objects;
  const int64_t* block_ids;  // slice base for this launch
  int total_blocks;          // num_objects * blocks_per_object
  int skip_prefix_n_blocks;  // within this launch
  bool lmcache_to_engine;    // TransferDirection::H2D
  int layout;
  PageBufferShapeDesc shape_desc;
  int lmcache_chunk_size;
};

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(const LaunchArgs& la,
                                             cudaStream_t stream) {
  const int num_objects = la.num_objects;
  PORT_CHECK(num_objects >= 1 && num_objects <= 4);
  const int total_blocks = la.total_blocks;
  PORT_CHECK(total_blocks % num_objects == 0);
  const int num_blocks_per_object = total_blocks / num_objects;
  const PageBufferShapeDesc shape_desc = la.shape_desc;
  PORT_CHECK(num_blocks_per_object * shape_desc.bs == la.lmcache_chunk_size);

  MemoryObj4<ScalarType> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects) ? reinterpret_cast<ScalarType*>(la.objects[i])
                          : nullptr;
  }
  ScalarType** paged_buffer_ptrs = reinterpret_cast<ScalarType**>(
      const_cast<uint64_t*>(la.paged_layer_ptrs));
  const int64_t* block_ids_ptr = la.block_ids;
  const int lmcache_chunk_size = la.lmcache_chunk_size;
  const int skip_prefix_n_blocks = la.skip_prefix_n_blocks;

  // --- Grid and block dimensions (upstream, verbatim arithmetic) ---
  int elements_per_head = shape_desc.hs * shape_desc.element_size /
                          static_cast<int>(sizeof(ScalarType));
  int thread_dim_x = elements_per_head < 32 ? elements_per_head : 32;
  int thread_dim_y = shape_desc.nh;
  PORT_CHECK(thread_dim_y <= 32);
  int thread_dim_z = shape_desc.bs < (1024 / (thread_dim_x * thread_dim_y))
                         ? shape_desc.bs
                         : (1024 / (thread_dim_x * thread_dim_y));
  thread_dim_z = thread_dim_z < 64 ? thread_dim_z : 64;

  dim3 block(thread_dim_x, thread_dim_y, thread_dim_z);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

#define LAUNCH_KERNEL(DIRECTION, FORMAT)                                    \
  multi_layer_block_transfer_kernel<ScalarType, DIRECTION, FORMAT>          \
      <<<grid, block, 0, stream>>>(lmcache_obj4, paged_buffer_ptrs,         \
                                   block_ids_ptr, num_blocks_per_object,    \
                                   shape_desc, lmcache_chunk_size,          \
                                   skip_prefix_n_blocks);

  if (la.layout == LMC_LAYOUT_SPLIT_HND) {
    if (la.lmcache_to_engine) {
      LAUNCH_KERNEL(true, LMC_LAYOUT_SPLIT_HND);
    } else {
      LAUNCH_KERNEL(false, LMC_LAYOUT_SPLIT_HND);
    }
  } else {
    if (la.lmcache_to_engine) {
      LAUNCH_KERNEL(true, LMC_LAYOUT_FUSED_HND);
    } else {
      LAUNCH_KERNEL(false, LMC_LAYOUT_FUSED_HND);
    }
  }
#undef LAUNCH_KERNEL
}

void multi_layer_block_kv_transfer(const LaunchArgs& la, cudaStream_t stream) {
  int head_bytes = la.shape_desc.hs * la.shape_desc.element_size;
  PORT_CHECK(head_bytes % 2 == 0);
  if (head_bytes % sizeof(uint4) == 0) {
    multi_layer_block_kv_transfer_templated<uint4>(la, stream);
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    multi_layer_block_kv_transfer_templated<uint32_t>(la, stream);
  } else {
    multi_layer_block_kv_transfer_templated<uint16_t>(la, stream);
  }
}

}  // namespace

// --- Staged executor: lmcache_driven_transfer._run_object_group_transfer_plan
// + mp_mem_kernels.cu execute_object_group_transfer, with the ABI glue noted
// in the header comment ------------------------------------------------------
extern "C" void lmcache_paged_kv_block_transfer(const TransferParams* p,
                                                cudaStream_t stream) {
  PORT_CHECK(p->layout == LMC_LAYOUT_SPLIT_HND ||
             p->layout == LMC_LAYOUT_FUSED_HND);
  PORT_CHECK(p->kv_size == (p->layout == LMC_LAYOUT_SPLIT_HND ? 2 : 1));
  const int num_objects = p->num_objects;
  PORT_CHECK(num_objects >= 1 && num_objects <= 4);
  const int bpo = p->blocks_per_object;
  PORT_CHECK(bpo * num_objects == p->total_blocks);
  PORT_CHECK(bpo * p->bs == p->chunk_tokens);

  LaunchArgs la;
  la.paged_layer_ptrs = p->paged_layer_ptrs;
  la.lmcache_to_engine = (p->direction == LMC_DIR_RETRIEVE);
  la.layout = p->layout;
  la.shape_desc.kv_size = p->kv_size;
  la.shape_desc.nl = p->nl;
  la.shape_desc.nb = p->nb;
  la.shape_desc.bs = p->bs;
  la.shape_desc.nh = p->nh;
  la.shape_desc.hs = p->hs;
  la.shape_desc.element_size = p->element_size;
  la.shape_desc.block_stride_elems = 0;
  la.lmcache_chunk_size = p->chunk_tokens;

  const int skip = p->skip_prefix_n_blocks;
  const size_t token_bytes =
      static_cast<size_t>(p->nh) * p->hs * p->element_size;
  const size_t slab_bytes = static_cast<size_t>(p->chunk_tokens) * token_bytes;
  const size_t obj_bytes = slab_bytes * p->kv_size * p->nl;  // 2LTD object
  const size_t arena = p->device_scratch && p->device_scratch_bytes > 0
                           ? static_cast<size_t>(p->device_scratch_bytes)
                           : 0;
  int slots = static_cast<int>(arena / obj_bytes);
  if (slots > 4) slots = 4;

  if (!p->chunk_side_is_host || slots < 1) {
    // Device-resident objects (or no room to stage): one direct launch.
    for (int i = 0; i < 4; ++i)
      la.objects[i] = (i < num_objects) ? p->chunk_ptrs[i] : 0;
    la.num_objects = num_objects;
    la.block_ids = p->block_ids;
    la.total_blocks = p->total_blocks;
    la.skip_prefix_n_blocks = skip;
    multi_layer_block_kv_transfer(la, stream);
    return;
  }

  char* temp = static_cast<char*>(p->device_scratch);
  // batched_iteration_with_skip(memory_objs, batch_size, skip_count=0)
  for (int start = 0; start < num_objects; start += slots) {
    const int batch_len =
        (start + slots < num_objects) ? slots : (num_objects - start);
    const int batch_start_block = start * bpo;
    const int batch_end_block = (start + batch_len) * bpo;
    const int effective_start = skip > batch_start_block ? skip : batch_start_block;
    if (effective_start >= batch_end_block) continue;  // whole batch skipped
    const int skip_in_batch = effective_start - batch_start_block;

    for (int i = 0; i < 4; ++i)
      la.objects[i] = (i < batch_len)
                          ? reinterpret_cast<uint64_t>(temp + static_cast<size_t>(i) * obj_bytes)
                          : 0;
    la.num_objects = batch_len;
    la.block_ids = p->block_ids + batch_start_block;
    la.total_blocks = batch_len * bpo;
    la.skip_prefix_n_blocks = skip_in_batch;

    if (la.lmcache_to_engine) {
      // H2D: build_staging_copies -> lmcache_memcpy_async (whole objects),
      // then the kernel reads the GPU temp buffers.
      for (int i = 0; i < batch_len; ++i) {
        cudaMemcpyAsync(temp + static_cast<size_t>(i) * obj_bytes,
                        reinterpret_cast<const void*>(p->chunk_ptrs[start + i]),
                        obj_bytes, cudaMemcpyHostToDevice, stream);
      }
      multi_layer_block_kv_transfer(la, stream);
    } else {
      // D2H: kernel writes the GPU temp buffers, then staging copies out.
      multi_layer_block_kv_transfer(la, stream);
      for (int i = 0; i < batch_len; ++i) {
        const int o = start + i;
        int s0 = skip - o * bpo;  // skipped blocks inside this object
        s0 = s0 < 0 ? 0 : (s0 > bpo ? bpo : s0);
        char* dev = temp + static_cast<size_t>(i) * obj_bytes;
        char* hst = reinterpret_cast<char*>(p->chunk_ptrs[o]);
        if (s0 == bpo) continue;  // ABI glue: fully skipped object untouched
        if (s0 == 0) {
          cudaMemcpyAsync(hst, dev, obj_bytes, cudaMemcpyDeviceToHost, stream);
        } else {
          // ABI glue: keep each slab's skipped token prefix untouched.
          const size_t off = static_cast<size_t>(s0) * p->bs * token_bytes;
          cudaMemcpy2DAsync(hst + off, slab_bytes, dev + off, slab_bytes,
                            slab_bytes - off,
                            static_cast<size_t>(p->kv_size) * p->nl,
                            cudaMemcpyDeviceToHost, stream);
        }
      }
    }
  }
}
