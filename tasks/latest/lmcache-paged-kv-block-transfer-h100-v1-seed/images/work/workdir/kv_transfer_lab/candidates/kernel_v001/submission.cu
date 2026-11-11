#include <type_traits>

#include "transfer_abi.h"

namespace {

#define CHECK_OR_RETURN(cond, msg) \
  do {                        \
    if (!(cond)) {            \
      return;                 \
    }                         \
  } while (0)

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;
};

struct ShapeDesc {
  int kv_size;
  int nl;
  int nb;
  int bs;
  int nh;
  int hs;
  int element_size;

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
    return static_cast<size_t>(bs) * nh * hs * element_size /
           sizeof(ScalarType);
  }
};

template <typename ScalarType, int format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const ShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == LMC_LAYOUT_SPLIT_HND) {
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else {
    return engine_block_idx * scalars_per_block;
  }
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx, const ShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_head_block = shape_desc.bs * scalars_per_head;
  return head_idx * scalars_per_head_block + token_offset * scalars_per_head;
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_chunk_global_offset(
    const int k_or_v, const int token_offset_in_chunk_object,
    const int layer_idx, const int chunk_chunk_size,
    const ShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return token_offset_in_chunk_object * scalars_per_token +
         layer_idx * chunk_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * chunk_chunk_size * scalars_per_token;
}

template <typename ScalarType, int format>
__device__ inline size_t calculate_chunk_local_offset(
    const int token_offset, const int head_idx, const ShapeDesc shape_desc) {
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

template <typename ScalarType, bool chunk_to_engine, int format>
__device__ void block_transfer_single_block(
    ScalarType* __restrict__ chunk_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_chunk_block, const ShapeDesc shape_desc,
    const int chunk_chunk_size) {
  const int head_idx = threadIdx.y;
  const int init_token_offset = threadIdx.z;
  const int token_stride = blockDim.z;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t chunk_global_offset =
      calculate_chunk_global_offset<ScalarType, format>(
          k_or_v, offset_in_chunk_block, layer_idx, chunk_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr =
      (ScalarType*)paged_buffer_ptrs[layer_idx];

  for (int token_offset = init_token_offset; token_offset < shape_desc.bs;
       token_offset += token_stride) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t chunk_local_offset =
        calculate_chunk_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* chunk_ptr =
        chunk_object + chunk_global_offset + chunk_local_offset;
    if constexpr (chunk_to_engine) {
      warp_copy<ScalarType>(engine_ptr, chunk_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(chunk_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool chunk_to_engine, int format>
__global__ void block_transfer_kernel(
    MemoryObj4<ScalarType> chunk_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids, const int num_blocks_per_object,
    const ShapeDesc shape_desc, const int chunk_chunk_size,
    const int skip_prefix_n_blocks) {
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;

  const int engine_block_idx = engine_block_ids[flat_block_idx];
  block_transfer_single_block<ScalarType, chunk_to_engine,
                                          format>(
      chunk_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs, shape_desc, chunk_chunk_size);
}

template <typename ScalarType>
void launch_templated(const TransferParams* p, cudaStream_t stream) {
  const int num_objects = p->num_objects;
  CHECK_OR_RETURN(num_objects >= 1 && num_objects <= 4, "num_objects in 1..4");
  const int total_blocks = p->total_blocks;
  CHECK_OR_RETURN(total_blocks % num_objects == 0, "total divisible by objects");
  const int num_blocks_per_object = total_blocks / num_objects;
  CHECK_OR_RETURN(num_blocks_per_object == p->blocks_per_object,
             "blocks_per_object consistent");
  CHECK_OR_RETURN(num_blocks_per_object * p->bs == p->chunk_tokens,
             "blocks_per_object * bs == chunk_tokens");

  ShapeDesc shape_desc;
  shape_desc.kv_size = p->kv_size;
  shape_desc.nl = p->nl;
  shape_desc.nb = p->nb;
  shape_desc.bs = p->bs;
  shape_desc.nh = p->nh;
  shape_desc.hs = p->hs;
  shape_desc.element_size = p->element_size;

  MemoryObj4<ScalarType> chunk_obj4;
  chunk_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    chunk_obj4.objects[i] =
        (i < num_objects) ? reinterpret_cast<ScalarType*>(p->chunk_ptrs[i])
                          : nullptr;
  }

  ScalarType** paged_buffer_ptrs = reinterpret_cast<ScalarType**>(
      const_cast<uint64_t*>(p->paged_layer_ptrs));
  const int64_t* block_ids_ptr = p->block_ids;

  int elements_per_head = shape_desc.hs * shape_desc.element_size /
                          static_cast<int>(sizeof(ScalarType));
  int thread_dim_x = elements_per_head < 32 ? elements_per_head : 32;
  int thread_dim_y = shape_desc.nh;
  CHECK_OR_RETURN(thread_dim_y <= 32, "nh <= 32");
  int thread_dim_z = shape_desc.bs < (1024 / (thread_dim_x * thread_dim_y))
                         ? shape_desc.bs
                         : (1024 / (thread_dim_x * thread_dim_y));
  thread_dim_z = thread_dim_z < 64 ? thread_dim_z : 64;

  dim3 block(thread_dim_x, thread_dim_y, thread_dim_z);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

#define PORT_LAUNCH(DIRECTION, FORMAT)                                     \
  block_transfer_kernel<ScalarType, DIRECTION, FORMAT>         \
      <<<grid, block, 0, stream>>>(chunk_obj4, paged_buffer_ptrs,        \
                                   block_ids_ptr, num_blocks_per_object,   \
                                   shape_desc, p->chunk_tokens,            \
                                   p->skip_prefix_n_blocks);               \
  CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, "kernel launch");

  const bool chunk_to_engine = (p->direction == LMC_DIR_RETRIEVE);
  if (p->layout == LMC_LAYOUT_SPLIT_HND) {
    CHECK_OR_RETURN(p->kv_size == 2, "split kv_size 2");
    if (chunk_to_engine) {
      PORT_LAUNCH(true, LMC_LAYOUT_SPLIT_HND);
    } else {
      PORT_LAUNCH(false, LMC_LAYOUT_SPLIT_HND);
    }
  } else if (p->layout == LMC_LAYOUT_FUSED_HND) {
    CHECK_OR_RETURN(p->kv_size == 1, "fused kv_size 1");
    if (chunk_to_engine) {
      PORT_LAUNCH(true, LMC_LAYOUT_FUSED_HND);
    } else {
      PORT_LAUNCH(false, LMC_LAYOUT_FUSED_HND);
    }
  } else {
    CHECK_OR_RETURN(false, "unsupported layout");
  }
#undef PORT_LAUNCH
}

}

extern "C" void lmcache_paged_kv_block_transfer(const TransferParams* p,
                                                cudaStream_t stream) {
  int head_bytes = p->hs * p->element_size;
  CHECK_OR_RETURN(head_bytes % 2 == 0, "head bytes divisible by 2");
  if (head_bytes % sizeof(uint4) == 0) {
    launch_templated<uint4>(p, stream);
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    launch_templated<uint32_t>(p, stream);
  } else {
    launch_templated<uint16_t>(p, stream);
  }
}
