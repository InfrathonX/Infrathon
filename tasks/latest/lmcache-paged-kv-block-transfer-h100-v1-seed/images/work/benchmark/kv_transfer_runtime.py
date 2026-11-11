#!/usr/bin/env python3
"""Public test runtime for the paged-KV block transfer task."""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
ENTRYPOINT = 'lmcache_paged_kv_block_transfer'
BS = 16
HEAD_SIZES = (64, 128)
ESIZE = 2
POISON = 31868
SCRATCH_BYTES = 256 * 1024 * 1024
SETTLE_SECONDS = 0.1
DEVICE_SETTLE_SECONDS = 0.05
DIR_STORE = 0
DIR_RETRIEVE = 1
LAYOUT_SPLIT = 0
LAYOUT_FUSED = 1
LAYOUT_NAMES = {LAYOUT_SPLIT: 'split', LAYOUT_FUSED: 'fused'}
LAYOUT_IDS = {name: value for value, name in LAYOUT_NAMES.items()}
DIRECTION_NAMES = {DIR_STORE: 'store', DIR_RETRIEVE: 'retrieve'}
DIRECTION_IDS = {name: value for value, name in DIRECTION_NAMES.items()}
NL_RANGE = (1, 128)
NH_RANGE = (1, 32)
NB_RANGE = (1, 65536)
OBJECTS_RANGE = (1, 4)
DEFAULT_TRIALS = 3
GROUP_WEIGHTS = {'device': 0.35, 'host': 0.25, 'trace': 0.1, 'overlap': 0.3}
OVERLAP_COVER = 3.0
MAX_OVERLAP_LOSS = 0.95
RECORD_CONSTANTS = {'bs': BS, 'head_sizes': list(HEAD_SIZES), 'esize': ESIZE, 'poison': POISON, 'scratch_bytes': SCRATCH_BYTES, 'small_call_bytes': 16 * 1024 * 1024, 'overlap_cover': OVERLAP_COVER, 'max_overlap_loss': MAX_OVERLAP_LOSS}
NVCC = '/usr/local/cuda/bin/nvcc'

def native_arch(torch: Any) -> str:
    major, minor = (int(v) for v in torch.cuda.get_device_capability(0))
    return f'sm_{major}{minor}' + ('a' if major >= 9 else '')

def compile_argv(source: Path, output: Path, include_dir: Path, arch: str) -> list[str]:
    if not re.fullmatch('sm_[0-9]{2,3}a?', arch):
        raise ValueError(f'unexpected nvcc architecture {arch!r}')
    return [NVCC, '-O3', '-std=c++17', f'-arch={arch}', '--shared', '--cudart=shared', '-Xcompiler', '-fPIC', f'-I{include_dir}', '-o', str(output), str(source)]

def compile_candidate(source: Path, out_dir: Path, include_dir: Path, *, arch: str, timeout: float=300.0) -> tuple[Path, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / 'submission.so'
    process = subprocess.run(compile_argv(Path(source), output, Path(include_dir), arch), capture_output=True, text=True, timeout=timeout, check=False)
    log = (process.stdout or '') + (process.stderr or '')
    if process.returncode != 0 or not output.is_file():
        raise RuntimeError(f'candidate compilation failed:\n{log[-4000:]}')
    return (output, log)
ALLOWED_LIBC_SYMBOLS = frozenset({'__cxa_atexit', '__cxa_finalize', '__gmon_start__', '_ITM_registerTMCloneTable', '_ITM_deregisterTMCloneTable', 'abort', 'free', 'malloc', 'memcpy', 'memmove', 'memset', 'strlen', '__stack_chk_fail', '__assert_fail', '__gxx_personality_v0', '_Unwind_Resume'})
ALLOWED_CUDA_SYMBOLS = frozenset({'cudaLaunchKernel', 'cudaLaunchKernelExC', 'cudaLaunchCooperativeKernel', 'cudaMemcpyAsync', 'cudaMemcpy2DAsync', 'cudaMemcpy3DAsync', 'cudaMemsetAsync', 'cudaMemset2DAsync', 'cudaGetLastError', 'cudaPeekAtLastError', 'cudaGetErrorString', 'cudaGetErrorName', 'cudaFuncSetAttribute', 'cudaFuncGetAttributes', 'cudaFuncSetCacheConfig', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor', 'cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', 'cudaDeviceGetAttribute', 'cudaGetDevice', 'cudaGetDeviceProperties', 'cudaGetDeviceProperties_v2'})
ALLOWED_UNDEFINED_PREFIXES = ('__cuda', '_ZSt', '_ZNSt', '_Znwm', '_ZdlPv')

def inspect_binary(library: Path) -> dict[str, Any]:
    process = subprocess.run(['readelf', '--dyn-syms', '--wide', str(library)], capture_output=True, text=True, timeout=60.0, check=True)
    undefined: list[str] = []
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[6] == 'UND' and parts[7]:
            name = parts[7].split('@', 1)[0]
            if name:
                undefined.append(name)
    problems: list[str] = []
    for name in sorted(set(undefined)):
        if name in ALLOWED_LIBC_SYMBOLS or name in ALLOWED_CUDA_SYMBOLS:
            continue
        if name.startswith(ALLOWED_UNDEFINED_PREFIXES):
            continue
        problems.append(f'undefined symbol outside the allowlist: {name}')
    return {'ok': not problems, 'undefined_symbols': sorted(set(undefined)), 'problems': problems}

class TransferParams(ctypes.Structure):
    _fields_ = [('paged_layer_ptrs', ctypes.c_uint64), ('chunk_ptrs', ctypes.c_uint64 * 4), ('block_ids', ctypes.c_uint64), ('num_objects', ctypes.c_int32), ('total_blocks', ctypes.c_int32), ('blocks_per_object', ctypes.c_int32), ('skip_prefix_n_blocks', ctypes.c_int32), ('direction', ctypes.c_int32), ('layout', ctypes.c_int32), ('kv_size', ctypes.c_int32), ('nl', ctypes.c_int32), ('nb', ctypes.c_int32), ('bs', ctypes.c_int32), ('nh', ctypes.c_int32), ('hs', ctypes.c_int32), ('element_size', ctypes.c_int32), ('chunk_tokens', ctypes.c_int32), ('chunk_side_is_host', ctypes.c_int32), ('_pad0', ctypes.c_int32), ('device_scratch', ctypes.c_uint64), ('device_scratch_bytes', ctypes.c_int64)]
assert ctypes.sizeof(TransferParams) == 128, ctypes.sizeof(TransferParams)

@dataclass(frozen=True)
class Clock:
    perf_counter: Callable[[], float]
    synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]
    event: Callable[..., Any]

    @classmethod
    def capture(cls, torch: Any) -> 'Clock':
        return cls(perf_counter=time.perf_counter, synchronize=torch.cuda.synchronize, current_stream=torch.cuda.current_stream, event=torch.cuda.Event)

def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f'{label} is not a finite positive number: {value!r}')
    return number

def geometric_mean(values: Iterable[float]) -> float:
    items = [finite_positive(value, 'geometric mean term') for value in values]
    if not items:
        raise RuntimeError('geometric mean of an empty sequence')
    return math.exp(math.fsum((math.log(item) for item in items)) / len(items))
WAVE_KEYS = {'layout', 'nl', 'nh', 'hs', 'chunk_tokens', 'num_objects', 'direction', 'skip', 'nb', 'ids'}
CASE_KEYS = {'case_id', 'family', 'arm', 'foreground', 'stratum', 'tier', 'waves', 'trials', 'seed'}
POOL_KEYS = ('layout', 'nl', 'nh', 'hs', 'nb')
DIRECT_KEYS = ('layout', 'nl', 'nh', 'hs', 'chunk_tokens', 'num_objects', 'direction', 'nb')
DUPLEX_KEYS = ('layout', 'nl', 'nh', 'hs', 'chunk_tokens', 'num_objects', 'nb')
ARMS = ('device', 'host', 'trace', 'overlap', 'duplex')
FAMILIES = {'device-large': 'device', 'device-small': 'device', 'host-large': 'host', 'host-small': 'host', 'host-duplex': 'duplex', 'trace': 'trace', 'overlap': 'overlap'}
GROUP_FAMILIES = {'device': ('device-large', 'device-small'), 'host': ('host-large', 'host-small', 'host-duplex'), 'trace': ('trace',), 'overlap': ('overlap',)}
TIERS = ('paper', 'other', 'public', 'edge')
STRATUM_RE = re.compile('^[a-z0-9][a-z0-9-]{0,40}$')
SMALL_CALL_BYTES = 16 * 1024 * 1024
FOREGROUNDS = ('gemm', 'hbm')
MAX_DIRECT_WAVES = 128
MAX_TRACE_WAVES = 64
MAX_CASE_PAYLOAD_BYTES = 4 * 1024 ** 3
MAX_POOL_BYTES = 44 * 1000 ** 3
HOST_ARENA_BYTES = 4 * 1024 ** 3 + 64 * 1024 ** 2
ARENA_ALIGN = 2 * 1024 * 1024

def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{label} must be an integer')
    return int(value)

def validate_wave(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != WAVE_KEYS:
        raise ValueError(f'wave fields differ from schema: {sorted(raw)}')
    layout = str(raw['layout'])
    if layout not in LAYOUT_IDS:
        raise ValueError(f'unknown layout {layout!r}')
    direction = str(raw['direction'])
    if direction not in DIRECTION_IDS:
        raise ValueError(f'unknown direction {direction!r}')
    nl = _int(raw['nl'], 'nl')
    nh = _int(raw['nh'], 'nh')
    hs = _int(raw['hs'], 'hs')
    nb = _int(raw['nb'], 'nb')
    chunk_tokens = _int(raw['chunk_tokens'], 'chunk_tokens')
    num_objects = _int(raw['num_objects'], 'num_objects')
    skip = _int(raw['skip'], 'skip')
    if not NL_RANGE[0] <= nl <= NL_RANGE[1]:
        raise ValueError(f'nl outside {NL_RANGE}: {nl}')
    if not NH_RANGE[0] <= nh <= NH_RANGE[1]:
        raise ValueError(f'nh outside {NH_RANGE}: {nh}')
    if hs not in HEAD_SIZES:
        raise ValueError(f'hs outside {HEAD_SIZES}: {hs}')
    if not NB_RANGE[0] <= nb <= NB_RANGE[1]:
        raise ValueError(f'nb outside {NB_RANGE}: {nb}')
    if not OBJECTS_RANGE[0] <= num_objects <= OBJECTS_RANGE[1]:
        raise ValueError(f'num_objects outside {OBJECTS_RANGE}: {num_objects}')
    if chunk_tokens <= 0 or chunk_tokens % BS != 0:
        raise ValueError(f'chunk_tokens must be a positive multiple of {BS}')
    total_blocks = num_objects * (chunk_tokens // BS)
    if total_blocks > nb:
        raise ValueError('total blocks exceed the paged pool')
    if not 0 <= skip <= total_blocks:
        raise ValueError('skip outside [0, total_blocks]')
    ids = raw['ids']
    if not isinstance(ids, list) or len(ids) != total_blocks:
        raise ValueError('ids must list exactly one block id per flat block')
    ids = [_int(value, 'block id') for value in ids]
    if min(ids) < 0 or max(ids) >= nb or len(set(ids)) != len(ids):
        raise ValueError('block ids must be unique and inside the pool')
    return {'layout': layout, 'nl': nl, 'nh': nh, 'hs': hs, 'chunk_tokens': chunk_tokens, 'num_objects': num_objects, 'direction': direction, 'skip': skip, 'nb': nb, 'ids': ids}

def wave_geometry(wave: Mapping[str, Any]) -> dict[str, int]:
    layout_id = LAYOUT_IDS[str(wave['layout'])]
    kv = 2 if layout_id == LAYOUT_SPLIT else 1
    width = int(wave['hs']) if layout_id == LAYOUT_SPLIT else 2 * int(wave['hs'])
    blocks_per_object = int(wave['chunk_tokens']) // BS
    total_blocks = int(wave['num_objects']) * blocks_per_object
    return {'layout_id': layout_id, 'kv': kv, 'hs': width, 'hidden': int(wave['nh']) * width, 'blocks_per_object': blocks_per_object, 'total_blocks': total_blocks}

def wave_logical_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    moved_blocks = geometry['total_blocks'] - int(wave['skip'])
    return moved_blocks * geometry['kv'] * int(wave['nl']) * BS * int(wave['nh']) * geometry['hs'] * ESIZE

def wave_object_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    return geometry['kv'] * int(wave['nl']) * int(wave['chunk_tokens']) * geometry['hidden'] * ESIZE

def pool_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    return geometry['kv'] * int(wave['nl']) * int(wave['nb']) * int(wave['nh']) * BS * geometry['hs'] * ESIZE

def case_payload_bytes(case: Mapping[str, Any]) -> int:
    return sum((wave_object_bytes(wave) * int(wave['num_objects']) for wave in case['waves']))

def case_logical_bytes(case: Mapping[str, Any]) -> int:
    return sum((wave_logical_bytes(wave) for wave in case['waves']))

def case_host_chunks(case: Mapping[str, Any]) -> bool:
    return str(case['arm']) in ('host', 'trace', 'overlap', 'duplex')

def validate_case(raw: Mapping[str, Any], *, scored: bool) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - {'scored'} != CASE_KEYS:
        raise ValueError(f'case fields differ from schema: {sorted(raw)}')
    case_id = str(raw['case_id'])
    family = str(raw['family'])
    arm = str(raw['arm'])
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    if family not in FAMILIES or FAMILIES[family] != arm:
        raise ValueError(f'case {case_id}: family {family!r} does not belong to arm {arm!r}')
    foreground = raw['foreground']
    if arm == 'overlap':
        if foreground not in FOREGROUNDS:
            raise ValueError(f'case {case_id}: overlap needs a foreground in {FOREGROUNDS}')
    elif foreground is not None:
        raise ValueError(f'case {case_id}: only overlap cases name a foreground')
    waves_raw = raw['waves']
    if not isinstance(waves_raw, list) or not waves_raw:
        raise ValueError(f'case {case_id} has no waves')
    waves = [validate_wave(wave) for wave in waves_raw]
    keys = POOL_KEYS if arm == 'trace' else DUPLEX_KEYS if arm == 'duplex' else DIRECT_KEYS
    if len({tuple((wave[key] for key in keys)) for wave in waves}) != 1:
        raise ValueError(f'case {case_id}: waves must share {keys}')
    if arm == 'duplex':
        stores = {b for w in waves if w['direction'] == 'store' for b in w['ids']}
        loads = {b for w in waves if w['direction'] == 'retrieve' for b in w['ids']}
        if not stores or not loads:
            raise ValueError(f'case {case_id}: a duplex case needs stores and retrieves')
        if stores & loads:
            raise ValueError(f'case {case_id}: duplex stores and retrieves must use disjoint blocks')
    stratum = str(raw['stratum'])
    tier = str(raw['tier'])
    if STRATUM_RE.fullmatch(stratum) is None or tier not in TIERS:
        raise ValueError(f'case {case_id}: stratum or tier differs from the schema')
    limit = MAX_TRACE_WAVES if arm == 'trace' else MAX_DIRECT_WAVES
    if len(waves) > limit:
        raise ValueError(f'case {case_id} exceeds {limit} waves')
    trials = _int(raw['trials'], 'trials')
    if not 1 <= trials <= 32:
        raise ValueError('trials outside [1, 32]')
    seed = _int(raw['seed'], 'seed')
    case = {'case_id': case_id, 'family': family, 'arm': arm, 'foreground': foreground, 'stratum': stratum, 'tier': tier, 'waves': waves, 'trials': trials, 'seed': seed, 'scored': bool(scored)}
    if case_payload_bytes(case) > MAX_CASE_PAYLOAD_BYTES:
        raise ValueError(f'case {case_id}: chunk payload exceeds {MAX_CASE_PAYLOAD_BYTES} bytes')
    if case_host_chunks(case) and case_payload_bytes(case) > HOST_ARENA_BYTES - 4 * ARENA_ALIGN * len(waves):
        raise ValueError(f'case {case_id}: host chunks exceed the pinned arena')
    if pool_bytes(waves[0]) > MAX_POOL_BYTES:
        raise ValueError(f'case {case_id}: paged pool exceeds {MAX_POOL_BYTES} bytes')
    if scored and family in ('device-large', 'device-small', 'host-large', 'host-small'):
        small = family.endswith('-small')
        for wave in waves:
            if (wave_logical_bytes(wave) < SMALL_CALL_BYTES) != small:
                raise ValueError(f"case {case_id}: a wave's size does not match family {family}")
    return case

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_candidate(library_path: Path) -> Callable[..., None]:
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, ENTRYPOINT)
    function.argtypes = [ctypes.POINTER(TransferParams), ctypes.c_void_p]
    function.restype = None
    return function
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024 ** 3

def _pcie_provenance(index: int) -> dict[str, Any]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return {'pcie_link_gen': int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)), 'pcie_link_width': int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))}
    except Exception:
        return {'pcie_link_gen': None, 'pcie_link_width': None}

def hardware_inventory(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError('no CUDA device is visible')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'exactly one CUDA device must be visible, found {torch.cuda.device_count()}')
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    capability = (int(properties.major), int(properties.minor))
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(f'GPU compute capability {capability} is below the task minimum {MIN_COMPUTE_CAPABILITY}: {properties.name}')
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(f'GPU memory {total_memory_bytes} bytes is below the task minimum {MIN_DEVICE_MEMORY_BYTES} bytes: {properties.name}')
    return {'device_name': str(properties.name), 'uuid': str(getattr(properties, 'uuid', '')), 'capability': f'sm_{capability[0]}{capability[1]}', 'compute_capability': list(capability), 'min_compute_capability': list(MIN_COMPUTE_CAPABILITY), 'min_device_memory_bytes': MIN_DEVICE_MEMORY_BYTES, 'total_memory_bytes': total_memory_bytes, **_pcie_provenance(index), 'multi_processor_count': int(properties.multi_processor_count), 'torch_version': str(torch.__version__), 'cuda_version': str(torch.version.cuda)}

def _derive_seed(*parts: int) -> int:
    digest = hashlib.sha256(b'lmcache-kv-transfer-v2:' + b':'.join((str(part).encode() for part in parts))).digest()
    return int.from_bytes(digest[:8], 'little') & 9223372036854775807

def _poison_scratch(torch: Any, scratch: Any, generator: Any) -> None:
    scratch.random_(0, 256, generator=generator)

def _generator(torch: Any, device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

class HostArena:

    def __init__(self, torch: Any, nbytes: int) -> None:
        self.torch = torch
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.offset = 0

    def reset(self) -> None:
        self.offset = 0

    def take(self, shape: Sequence[int]) -> Any:
        count = 1
        for dim in shape:
            count *= int(dim)
        nbytes = count * ESIZE
        start = (self.offset + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN
        if start + nbytes > self.buffer.numel():
            raise RuntimeError('host chunks exceed the pinned arena')
        self.offset = start + nbytes
        return self.buffer[start:start + nbytes].view(self.torch.int16).view(*shape)

def _object_rows(torch: Any, geometry: Mapping[str, int], objects: Sequence[Any], layer: int, nh: int) -> Any:
    kv, width = (geometry['kv'], geometry['hs'])
    rows = torch.cat([obj[:, layer].reshape(kv, geometry['blocks_per_object'], BS, nh, width) for obj in objects], dim=1)
    return rows.permute(0, 1, 3, 2, 4)

def _select_blocks(layer_tensor: Any, layout_id: int, ids: Any) -> Any:
    if layout_id == LAYOUT_SPLIT:
        return layer_tensor[:, ids]
    return layer_tensor[ids].unsqueeze(0)

class PoolCase:

    def __init__(self, torch: Any, case: Mapping[str, Any], device: Any, arena: HostArena | None) -> None:
        self.torch = torch
        self.device = device
        self.case = case
        self.waves = list(case['waves'])
        first = self.waves[0]
        self.geometry0 = wave_geometry(first)
        self.nl, self.nh, self.nb = (int(first['nl']), int(first['nh']), int(first['nb']))
        kv, width = (self.geometry0['kv'], self.geometry0['hs'])
        if self.geometry0['layout_id'] == LAYOUT_SPLIT:
            self.pool_shape = (kv, self.nb, self.nh, BS, width)
        else:
            self.pool_shape = (self.nb, self.nh, BS, width)
        self.pool = [torch.empty(self.pool_shape, dtype=torch.int16, device=device) for _ in range(self.nl)]
        self.ptr_array = torch.tensor([t.data_ptr() for t in self.pool], dtype=torch.int64, device=device)
        self.host_chunks = case_host_chunks(case)
        if self.host_chunks and arena is None:
            raise RuntimeError('host chunks need the pinned arena')
        if arena is not None:
            arena.reset()
        self.chunks: list[list[Any]] = []
        self.base_ids: list[Any] = []
        self.ids: list[Any] = []
        largest = 0
        for wave in self.waves:
            geometry = wave_geometry(wave)
            shape = (geometry['kv'], self.nl, int(wave['chunk_tokens']), geometry['hidden'])
            objects = []
            for _ in range(int(wave['num_objects'])):
                if self.host_chunks:
                    objects.append(arena.take(shape))
                else:
                    objects.append(torch.empty(shape, dtype=torch.int16, device=device))
            largest = max(largest, wave_object_bytes(wave) // ESIZE)
            self.chunks.append(objects)
            base = torch.tensor(wave['ids'], dtype=torch.int64, device=device)
            self.base_ids.append(base)
            self.ids.append(torch.empty_like(base))
        self.staging = torch.empty(largest, dtype=torch.int16, device=device) if self.host_chunks else None
        self.seed = 0

    def _layer_seed(self, layer: int) -> int:
        return _derive_seed(self.seed, 2577, layer)

    def _object_seed(self, wave: int, obj: int) -> int:
        return _derive_seed(self.seed, 192, wave, obj)

    def _fill_layer(self, target: Any, layer: int) -> None:
        target.random_(-32768, 32768, generator=_generator(self.torch, self.device, self._layer_seed(layer)))

    def _source_object(self, wave: int, obj: int, shape: Sequence[int]) -> Any:
        out = self.torch.empty(tuple(shape), dtype=self.torch.int16, device=self.device)
        out.random_(-32768, 32768, generator=_generator(self.torch, self.device, self._object_seed(wave, obj)))
        return out

    def refresh(self, seed: int) -> None:
        torch = self.torch
        self.seed = int(seed)
        for layer, tensor in enumerate(self.pool):
            self._fill_layer(tensor, layer)
        rotation = _derive_seed(self.seed, 2827) % self.nb
        for base, ids in zip(self.base_ids, self.ids):
            ids.copy_(torch.remainder(base + rotation, self.nb))
        for index, wave in enumerate(self.waves):
            store = DIRECTION_IDS[str(wave['direction'])] == DIR_STORE
            for obj, tensor in enumerate(self.chunks[index]):
                if not self.host_chunks:
                    if store:
                        tensor.fill_(POISON)
                    else:
                        tensor.random_(-32768, 32768, generator=_generator(torch, self.device, self._object_seed(index, obj)))
                    continue
                staging = self.staging[:tensor.numel()].view(tensor.shape)
                if store:
                    staging.fill_(POISON)
                else:
                    staging.random_(-32768, 32768, generator=_generator(torch, self.device, self._object_seed(index, obj)))
                tensor.copy_(staging)
        torch.cuda.synchronize(self.device)

    def table_digest(self) -> str:
        tables = hashlib.sha256()
        for ids in self.ids:
            tables.update(ids.cpu().numpy().tobytes())
        tables.update(self.ptr_array.cpu().numpy().tobytes())
        return tables.hexdigest()

    def params_list(self, scratch: Any, retrieve_scratch: Any=None) -> list[TransferParams]:
        packed = []
        for index, wave in enumerate(self.waves):
            arena = scratch
            if retrieve_scratch is not None and wave['direction'] == 'retrieve':
                arena = retrieve_scratch
            geometry = wave_geometry(wave)
            params = TransferParams()
            params.paged_layer_ptrs = self.ptr_array.data_ptr()
            chunks = self.chunks[index]
            for slot in range(4):
                params.chunk_ptrs[slot] = chunks[slot].data_ptr() if slot < len(chunks) else 0
            params.block_ids = self.ids[index].data_ptr()
            params.num_objects = int(wave['num_objects'])
            params.total_blocks = geometry['total_blocks']
            params.blocks_per_object = geometry['blocks_per_object']
            params.skip_prefix_n_blocks = int(wave['skip'])
            params.direction = DIRECTION_IDS[str(wave['direction'])]
            params.layout = geometry['layout_id']
            params.kv_size = geometry['kv']
            params.nl = self.nl
            params.nb = self.nb
            params.bs = BS
            params.nh = self.nh
            params.hs = geometry['hs']
            params.element_size = ESIZE
            params.chunk_tokens = int(wave['chunk_tokens'])
            params.chunk_side_is_host = 1 if self.host_chunks else 0
            params.device_scratch = arena.data_ptr()
            params.device_scratch_bytes = int(arena.numel())
            packed.append(params)
        return packed

    def verify(self) -> int:
        torch = self.torch
        device = self.device
        bad = 0
        sources: dict[int, list[Any]] = {}
        stored: dict[int, list[Any]] = {}
        for index, wave in enumerate(self.waves):
            geometry = wave_geometry(wave)
            shape = (geometry['kv'], self.nl, int(wave['chunk_tokens']), geometry['hidden'])
            if DIRECTION_IDS[str(wave['direction'])] == DIR_RETRIEVE:
                expected_objects = []
                for obj, tensor in enumerate(self.chunks[index]):
                    want = self._source_object(index, obj, shape)
                    got = tensor.to(device, non_blocking=False)
                    if not torch.equal(got, want):
                        bad += int((got != want).sum().item())
                    expected_objects.append(want)
                sources[index] = expected_objects
            else:
                stored[index] = [tensor.to(device) for tensor in self.chunks[index]]
        expected = torch.empty(self.pool_shape, dtype=torch.int16, device=device)
        layout_id = self.geometry0['layout_id']
        for layer in range(self.nl):
            self._fill_layer(expected, layer)
            for index, wave in enumerate(self.waves):
                geometry = wave_geometry(wave)
                ids = self.ids[index]
                skip = int(wave['skip'])
                if index in sources:
                    rows = _object_rows(torch, geometry, sources[index], layer, self.nh)
                    if skip < geometry['total_blocks']:
                        if layout_id == LAYOUT_SPLIT:
                            expected[:, ids[skip:]] = rows[:, skip:]
                        else:
                            expected[ids[skip:]] = rows[0, skip:]
                    continue
                selected = _select_blocks(expected, layout_id, ids)
                want = selected.permute(0, 1, 3, 2, 4).reshape(geometry['kv'], geometry['total_blocks'] * BS, geometry['hidden'])
                if skip:
                    want = want.clone()
                    want[:, :skip * BS] = POISON
                bpo = geometry['blocks_per_object']
                for obj, got in enumerate(stored[index]):
                    piece = want[:, obj * bpo * BS:(obj + 1) * bpo * BS]
                    have = got[:, layer]
                    if not torch.equal(have, piece):
                        bad += int((have != piece).sum().item())
            if not torch.equal(self.pool[layer], expected):
                bad += int((self.pool[layer] != expected).sum().item())
        del expected, sources, stored
        return bad

class Foreground:

    def __init__(self, torch: Any, kind: str, device: Any) -> None:
        self.torch = torch
        self.kind = kind
        self.stream = torch.cuda.Stream(device)
        if kind == 'gemm':
            self.a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
            self.b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
            self.c = torch.empty(8192, 8192, dtype=torch.bfloat16, device=device)
        elif kind == 'hbm':
            self.src = torch.empty(1024 ** 3, dtype=torch.uint8, device=device)
            self.src.random_(0, 256)
            self.dst = torch.empty_like(self.src)
        else:
            raise ValueError(f'unknown foreground {kind!r}')

    def step(self) -> None:
        if self.kind == 'gemm':
            self.torch.mm(self.a, self.b, out=self.c)
        else:
            self.dst.copy_(self.src)

    def launch(self, clock: Clock, iterations: int) -> tuple[Any, Any]:
        start = clock.event(enable_timing=True)
        end = clock.event(enable_timing=True)
        with self.torch.cuda.stream(self.stream):
            start.record(self.stream)
            for _ in range(iterations):
                self.step()
            end.record(self.stream)
        return (start, end)

    def alone(self, clock: Clock, iterations: int) -> float:
        clock.synchronize()
        start, end = self.launch(clock, iterations)
        clock.synchronize()
        return finite_positive(float(start.elapsed_time(end)) / 1000.0, 'foreground time')

def _invoke_params(function: Callable[..., None], packed: Sequence[TransferParams], stream_handle: int) -> None:
    handle = ctypes.c_void_p(stream_handle)
    for params in packed:
        function(ctypes.byref(params), handle)

class DuplexStreams:

    def __init__(self, torch: Any, device: Any) -> None:
        self.torch = torch
        self.store = torch.cuda.Stream(device)
        self.retrieve = torch.cuda.Stream(device)
        self.scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device)

def _launch(function: Callable[..., None], case_set: PoolCase, packed: Sequence[TransferParams], clock: Clock, duplex: DuplexStreams | None) -> None:
    stream = clock.current_stream()
    if duplex is None:
        _invoke_params(function, packed, stream.cuda_stream)
        return
    begin = clock.event()
    begin.record(stream)
    duplex.store.wait_event(begin)
    duplex.retrieve.wait_event(begin)
    stores = [p for p, w in zip(packed, case_set.waves) if w['direction'] == 'store']
    loads = [p for p, w in zip(packed, case_set.waves) if w['direction'] == 'retrieve']
    store_handle = ctypes.c_void_p(duplex.store.cuda_stream)
    load_handle = ctypes.c_void_p(duplex.retrieve.cuda_stream)
    for index in range(max(len(stores), len(loads))):
        if index < len(stores):
            function(ctypes.byref(stores[index]), store_handle)
        if index < len(loads):
            function(ctypes.byref(loads[index]), load_handle)
    stream.wait_stream(duplex.store)
    stream.wait_stream(duplex.retrieve)

def _settle(clock: Clock, spin: Any, seconds: float) -> None:
    deadline = clock.perf_counter() + seconds
    while clock.perf_counter() < deadline:
        for _ in range(50):
            spin.add_(1.0)
    clock.synchronize()

def _checked_pass(torch: Any, clock: Clock, function: Callable[..., None], case_set: PoolCase, scratch: Any, generator: Any, seed: int, label: str, duplex: DuplexStreams | None=None) -> None:
    case_set.refresh(seed)
    digest = case_set.table_digest()
    _poison_scratch(torch, scratch, generator)
    if duplex is not None:
        _poison_scratch(torch, duplex.scratch, generator)
    packed = case_set.params_list(scratch, duplex.scratch if duplex is not None else None)
    clock.synchronize()
    _launch(function, case_set, packed, clock, duplex)
    clock.synchronize()
    if case_set.table_digest() != digest:
        raise RuntimeError(f'{label}: block tables or pointer arrays were modified')
    bad = case_set.verify()
    if bad:
        raise RuntimeError(f'{label}: {bad} element(s) differ')

def _timed_pass(torch: Any, clock: Clock, function: Callable[..., None], case_set: PoolCase, scratch: Any, generator: Any, spin: Any, seed: int, label: str, foreground: Foreground | None=None, iterations: int=0, duplex: DuplexStreams | None=None) -> dict[str, Any]:
    case_set.refresh(seed)
    digest = case_set.table_digest()
    _poison_scratch(torch, scratch, generator)
    if duplex is not None:
        _poison_scratch(torch, duplex.scratch, generator)
    packed = case_set.params_list(scratch, duplex.scratch if duplex is not None else None)
    stream = clock.current_stream()
    start_event = clock.event(enable_timing=True)
    end_event = clock.event(enable_timing=True)
    clock.synchronize()
    _settle(clock, spin, SETTLE_SECONDS if case_set.host_chunks else DEVICE_SETTLE_SECONDS)
    fg_events = None
    if foreground is not None:
        fg_events = foreground.launch(clock, iterations)
    wall_start = clock.perf_counter()
    start_event.record(stream)
    _launch(function, case_set, packed, clock, duplex)
    end_event.record(stream)
    end_event.synchronize()
    host_wall = finite_positive(clock.perf_counter() - wall_start, 'host wall time')
    clock.synchronize()
    event_seconds = finite_positive(float(start_event.elapsed_time(end_event)) / 1000.0, 'CUDA event time')
    record: dict[str, Any] = {'host_wall': host_wall, 'event': event_seconds, 'seconds': max(host_wall, event_seconds)}
    if fg_events is not None:
        fg_start, fg_end = fg_events
        record['fg_seconds'] = finite_positive(float(fg_start.elapsed_time(fg_end)) / 1000.0, 'foreground time')
        record['fg_to_transfer_end'] = float(fg_start.elapsed_time(end_event)) / 1000.0
    if case_set.table_digest() != digest:
        raise RuntimeError(f'{label}: block tables or pointer arrays were modified')
    bad = case_set.verify()
    if bad:
        raise RuntimeError(f'{label}: {bad} element(s) differ')
    return record

def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))

def overlap_loss(fg_overlap: float, fg_alone: float, transfer: float) -> float:
    loss = (fg_overlap - fg_alone) / transfer
    return min(max(loss, 0.0), MAX_OVERLAP_LOSS)

def run_scored_case(torch: Any, clock: Clock, function: Callable[..., None], case: Mapping[str, Any], device: Any, scratch: Any, generator: Any, arena: HostArena | None) -> dict[str, Any]:
    case_id = str(case['case_id'])
    trials = int(case['trials'])
    seed = int(case['seed'])
    case_set = PoolCase(torch, case, device, arena)
    spin = torch.zeros(256, dtype=torch.float32, device=device)
    duplex = DuplexStreams(torch, device) if case['arm'] == 'duplex' else None
    _checked_pass(torch, clock, function, case_set, scratch, generator, _derive_seed(seed, 3), f'case {case_id} pre-call', duplex)
    logical = case_logical_bytes(case)
    row: dict[str, Any] = {'case_id': case_id, 'family': str(case['family']), 'arm': str(case['arm']), 'foreground': case['foreground'], 'stratum': str(case['stratum']), 'tier': str(case['tier']), 'scored': True, 'waves': len(case['waves']), 'trials': trials, 'logical_bytes': logical}
    foreground = None
    iterations = 0
    if case['arm'] == 'overlap':
        alone = [_timed_pass(torch, clock, function, case_set, scratch, generator, spin, _derive_seed(seed, 5, trial), f'case {case_id} alone {trial}') for trial in range(trials)]
        alone_seconds = [item['seconds'] for item in alone]
        foreground = Foreground(torch, str(case['foreground']), device)
        foreground.alone(clock, 3)
        per_step = foreground.alone(clock, 5) / 5.0
        iterations = max(4, int(math.ceil(OVERLAP_COVER * _median(alone_seconds) / per_step)))
        row['alone_seconds'] = alone_seconds
    records: list[dict[str, Any]] = []
    fg_alone: list[float] = []
    for trial in range(trials):
        label = f'case {case_id} trial {trial}'
        trial_seed = _derive_seed(seed, 4, trial + 1)
        if foreground is None:
            records.append(_timed_pass(torch, clock, function, case_set, scratch, generator, spin, trial_seed, label, duplex=duplex))
            continue
        for _attempt in range(4):
            record = _timed_pass(torch, clock, function, case_set, scratch, generator, spin, trial_seed, label, foreground, iterations)
            if record['fg_seconds'] >= record['fg_to_transfer_end']:
                break
            iterations *= 2
        else:
            raise RuntimeError(f'{label}: the foreground never covered the transfer')
        record['fg_iterations'] = iterations
        record['fg_alone'] = foreground.alone(clock, iterations)
        records.append(record)
        fg_alone.append(record['fg_alone'])
    elapsed = [item['seconds'] for item in records]
    median = _median(elapsed)
    gbps = logical / median / 1000000000.0
    row.update({'trial_seconds': elapsed, 'host_wall_seconds': [item['host_wall'] for item in records], 'event_seconds': [item['event'] for item in records], 'median_seconds': median, 'gbps': gbps})
    score = gbps
    if foreground is not None:
        losses = [overlap_loss(item['fg_seconds'], item['fg_alone'], item['seconds']) for item in records]
        row.update({'fg_overlap_seconds': [item['fg_seconds'] for item in records], 'fg_alone_seconds': fg_alone, 'fg_iterations': [item['fg_iterations'] for item in records], 'fg_loss': losses, 'fg_loss_median': _median(losses)})
        score = gbps * (1.0 - _median(losses))
        del foreground
    row['case_score'] = score
    del case_set
    torch.cuda.empty_cache()
    return row

def run_edge_case(torch: Any, clock: Clock, function: Callable[..., None], case: Mapping[str, Any], device: Any, scratch: Any, generator: Any, arena: HostArena | None) -> dict[str, Any]:
    case_set = PoolCase(torch, case, device, arena)
    duplex = DuplexStreams(torch, device) if case['arm'] == 'duplex' else None
    _checked_pass(torch, clock, function, case_set, scratch, generator, _derive_seed(int(case['seed']), 1), f"edge {case['case_id']}", duplex)
    del case_set
    torch.cuda.empty_cache()
    return {'case_id': str(case['case_id']), 'arm': str(case['arm']), 'scored': False, 'waves': len(case['waves']), 'correct': True}

def family_score(rows: Sequence[Mapping[str, Any]]) -> float:
    strata: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        strata.setdefault((str(row['tier']), str(row['stratum'])), []).append(float(row['case_score']))
    tiers: dict[str, list[float]] = {}
    for (tier, _stratum), values in strata.items():
        tiers.setdefault(tier, []).append(math.log(geometric_mean(values)))
    return math.exp(math.fsum((math.fsum(logs) / len(logs) for logs in tiers.values())) / len(tiers))

def aggregate(results: Sequence[Mapping[str, Any]], *, require_full_groups: bool=True) -> dict[str, Any]:
    scored = [row for row in results if row.get('scored')]
    if not scored:
        raise RuntimeError('no scored case produced a result')
    per_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored:
        per_family.setdefault(str(row['family']), []).append(row)
    family_scores = {name: family_score(rows) for name, rows in sorted(per_family.items())}
    groups: dict[str, float] = {}
    for group, families in GROUP_FAMILIES.items():
        present = [family_scores[name] for name in families if name in family_scores]
        if require_full_groups and len(present) != len(families):
            missing = [name for name in families if name not in family_scores]
            raise RuntimeError(f'no scored case for family(ies) {missing}')
        if present:
            groups[group] = geometric_mean(present)
    if require_full_groups:
        score = math.exp(math.fsum((GROUP_WEIGHTS[name] * math.log(groups[name]) for name in GROUP_WEIGHTS)))
    else:
        score = geometric_mean(groups.values())
    return {'family_scores': family_scores, 'group_scores': groups, 'group_weights': dict(GROUP_WEIGHTS), 'score_gbps': score, 'scored_cases': len(scored)}
PUBLIC_SUITE_ID = 'lmcache-kv-transfer-public-v3'
EDGE_SEED_BASE = 15532034

def load_case_table(path: Path, *, scored: bool=True) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 2:
        raise ValueError('case table schema differs')
    cases = [validate_case(case, scored=scored) for case in payload['cases']]
    if not cases:
        raise ValueError('case table contains no case')
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('case table contains duplicate case ids')
    return {'suite_id': str(payload['suite_id']), 'cases': cases}

def edge_ids(pattern: str, count: int, nb: int, seed: int) -> list[int]:
    import random as _random
    rng = _random.Random(seed)
    if pattern == 'contig':
        start = rng.randrange(0, nb - count + 1)
        return list(range(start, start + count))
    if pattern == 'reverse':
        start = rng.randrange(0, nb - count + 1)
        return list(range(start + count - 1, start - 1, -1))
    if pattern == 'ends':
        return [nb - 1 - i // 2 if i % 2 else i // 2 for i in range(count)]
    if pattern == 'runs':
        free = list(range(nb))
        rng.shuffle(free)
        ids: list[int] = []
        used: set[int] = set()
        for start in free:
            if len(ids) >= count:
                break
            length = rng.choice((1, 1, 2, 3, 5, 8))
            step = rng.choice((1, -1))
            run = [start + step * k for k in range(length)]
            run = [b for b in run if 0 <= b < nb and b not in used][:count - len(ids)]
            ids.extend(run)
            used.update(run)
        return ids
    return rng.sample(range(nb), count)

def _edge(case_id: str, arm: str, *, pattern: str='perm', waves: int=1, **overrides: Any) -> dict[str, Any]:
    base = {'layout': 'split', 'nl': 4, 'nh': 4, 'hs': 128, 'chunk_tokens': 64, 'num_objects': 2, 'direction': 'retrieve', 'skip': 0, 'nb': 64}
    base.update(overrides)
    count = base['num_objects'] * base['chunk_tokens'] // BS
    seed = _derive_seed(EDGE_SEED_BASE, int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], 'little'))
    wave_list = []
    for index in range(waves):
        wave = dict(base)
        wave['ids'] = edge_ids(pattern, count, base['nb'], _derive_seed(seed, index))
        wave_list.append(wave)
    if arm in ('trace', 'duplex'):
        for index, wave in enumerate(wave_list):
            wave['direction'] = 'store' if index % 2 else 'retrieve'
    if arm == 'duplex':
        half = base['nb'] // 2
        for wave in wave_list:
            offset = 0 if wave['direction'] == 'store' else half
            wave['ids'] = [offset + b % half for b in wave['ids']]
            if len(set(wave['ids'])) != len(wave['ids']):
                wave['ids'] = list(range(offset, offset + len(wave['ids'])))
    family = {'trace': 'trace', 'duplex': 'host-duplex'}.get(arm, f'{arm}-small')
    return validate_case({'case_id': case_id, 'family': family, 'arm': arm, 'foreground': None, 'stratum': 'edge', 'tier': 'edge', 'waves': wave_list, 'trials': 1, 'seed': seed}, scored=False)

def edge_fixtures() -> list[dict[str, Any]]:
    return [_edge('edge-skip-all', 'device', skip=8), _edge('edge-skip-almost-all', 'device', skip=7, direction='store'), _edge('edge-skip-partial-fused', 'device', layout='fused', skip=3), _edge('edge-single-block', 'device', chunk_tokens=16, num_objects=1), _edge('edge-three-objects', 'device', num_objects=3, chunk_tokens=128, nb=128), _edge('edge-tail-chunk', 'device', chunk_tokens=48, num_objects=1, direction='store'), _edge('edge-single-layer', 'device', nl=1), _edge('edge-deepest-stack', 'device', nl=128, nb=32, num_objects=1), _edge('edge-single-head', 'device', nh=1, pattern='reverse'), _edge('edge-most-heads', 'device', nh=32, nb=16, num_objects=1), _edge('edge-head-64-split', 'device', hs=64, pattern='runs', direction='store'), _edge('edge-head-64-fused', 'device', hs=64, layout='fused', pattern='runs'), _edge('edge-pool-ends', 'device', pattern='ends', nb=4096, direction='store'), _edge('edge-whole-pool', 'device', pattern='contig', nb=8), _edge('edge-repeated-blocks', 'device', pattern='contig', nb=8, waves=3), _edge('edge-largest-pool', 'device', layout='fused', nl=1, nh=1, hs=128, chunk_tokens=1024, num_objects=4, nb=65536, pattern='runs'), _edge('edge-object-beyond-scratch', 'device', nl=16, nh=8, chunk_tokens=4128, num_objects=1, nb=300), _edge('edge-host-single', 'host', num_objects=1, nh=3), _edge('edge-host-store-fused', 'host', layout='fused', direction='store'), _edge('edge-host-skip', 'host', skip=5, pattern='reverse'), _edge('edge-host-head-64', 'host', hs=64, nl=18, nh=2, direction='store'), _edge('edge-host-beyond-scratch', 'host', layout='fused', nl=16, nh=8, chunk_tokens=4224, num_objects=1, nb=512), _edge('edge-trace-overwrite', 'trace', pattern='contig', nb=8, waves=4), _edge('edge-duplex', 'duplex', pattern='perm', nb=128, waves=4), _edge('edge-duplex-skip-fused', 'duplex', layout='fused', skip=3, pattern='runs', nb=256, waves=6)]

def run_suite(torch: Any, library_path: Path, cases: Sequence[Mapping[str, Any]], *, suite_id: str, with_edge_fixtures: bool=True, require_full_groups: bool=True) -> dict[str, Any]:
    inventory = hardware_inventory(torch)
    clock = Clock.capture(torch)
    device = torch.device('cuda', torch.cuda.current_device())
    scratch = torch.empty(SCRATCH_BYTES, dtype=torch.uint8, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(_derive_seed(96774084))
    needs_arena = any((case_host_chunks(case) for case in cases)) or with_edge_fixtures
    arena = HostArena(torch, HOST_ARENA_BYTES) if needs_arena else None
    library_digest = sha256_file(Path(library_path))
    function = load_candidate(Path(library_path))
    results: list[dict[str, Any]] = []
    if with_edge_fixtures:
        for fixture in edge_fixtures():
            results.append(run_edge_case(torch, clock, function, fixture, device, scratch, generator, arena))
    for case in cases:
        results.append(run_scored_case(torch, clock, function, case, device, scratch, generator, arena))
    summary = aggregate(results, require_full_groups=require_full_groups)
    return {'suite_id': suite_id, 'library_sha256': library_digest, 'entrypoint': ENTRYPOINT, 'hardware': inventory, 'constants': RECORD_CONSTANTS, 'edge_fixtures': [row for row in results if not row.get('scored')], 'cases': [row for row in results if row.get('scored')], **summary}


def public_suite() -> list[dict[str, Any]]:
    return load_case_table(Path(__file__).resolve().parent / "public_cases.json")["cases"]
