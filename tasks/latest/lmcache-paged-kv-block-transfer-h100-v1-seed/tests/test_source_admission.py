"""Admission must accept the seed and fail closed on hostile constructs."""

from __future__ import annotations

import shutil

from conftest import TASK_ROOT, _load

admission = _load("kv_admission_t", "images/judge/source_admission.py")

BASE = (
    'extern "C" void lmcache_paged_kv_block_transfer('
    "const void* p, void* s) {}\n"
)

HOSTILE = {
    "cuda_malloc": BASE + "void f(){ void* p; cudaMalloc(&p, 8); }",
    "cuda_free": BASE + "void f(){ cudaFree(0); }",
    "host_alloc": BASE + "void f(){ void* p; cudaHostAlloc(&p, 8, 0); }",
    "host_register": BASE + "void f(){ cudaHostRegister(0, 0, 0); }",
    "stream_create": BASE + "void f(){ cudaStream_t s; cudaStreamCreate(&s); }",
    "event_create": BASE + "void f(){ cudaEvent_t e; cudaEventCreate(&e); }",
    "device_sync": BASE + "void f(){ cudaDeviceSynchronize(); }",
    "stream_sync": BASE + "void f(){ cudaStreamSynchronize(0); }",
    "sync_memcpy": BASE + "void f(){ cudaMemcpy(0, 0, 0, cudaMemcpyDefault); }",
    "sync_memset": BASE + "void f(){ cudaMemset(0, 0, 0); }",
    "peer_copy": BASE + "void f(){ cudaMemcpyPeerAsync(0, 0, 0, 0, 0, 0); }",
    "graphs": BASE + "void f(){ cudaGraph_t g; cudaGraphCreate(&g, 0); }",
    "capture": BASE + "void f(){ cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal); }",
    "host_func": BASE + "void f(){ cudaLaunchHostFunc(0, 0, 0); }",
    "driver_api": BASE + "void f(){ cuMemAlloc(0, 8); }",
    "dlopen": BASE + 'void f(){ dlopen("x", 1); }',
    "system": BASE + 'void f(){ system("id"); }',
    "fopen": BASE + 'void f(){ fopen("/etc/passwd", "r"); }',
    "getenv": BASE + 'void f(){ getenv("PATH"); }',
    "pthread": BASE + "void f(){ pthread_create(0, 0, 0, 0); }",
    "std_thread": BASE + "void f(){ std::thread t; }",
    "managed": BASE + "__managed__ int cache[16];",
    "bad_include": "#include <cstdio>\n" + BASE,
    "computed_include": '#include INDIRECT\n' + BASE,
    "x86_asm": BASE + 'void f(){ asm volatile("mov %%rax, %%rbx" ::: ); }',
    "syscall_asm": BASE + 'void f(){ asm volatile("syscall" ::: ); }',
    "int80_asm": BASE + 'void f(){ asm volatile("int $0x80" ::: ); }',
    "byte_asm": BASE + 'void f(){ asm volatile(".byte 0x0f, 0x05" ::: ); }',
    "pragma_gcc": BASE + "#pragma GCC poison foo\n",
    "underscore_pragma": BASE + '_Pragma("GCC poison foo")\n',
    "no_entrypoint": "void other() {}\n",
    "proc_path": BASE + 'const void* p2 = (const void*)"/proc/self/mem";',
}

BENIGN = {
    "seed": (TASK_ROOT / "oracle/candidates/kernel_v001/submission.cu").read_text(),
    "comment_mention": BASE + "// cudaMalloc is forbidden and this comment is fine\n",
    "string_mention": BASE + 'const char* note = "cudaMalloc";\n',
    "ptx_asm": BASE + (
        "__device__ void g(unsigned* d, const unsigned* s) {\n"
        '  asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(*d) : "l"(s));\n'
        "}\n"
    ),
    "memcpy_async": BASE + (
        "void h(void* d, const void* s, unsigned long n, cudaStream_t st) {\n"
        "  cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToDevice, st);\n"
        "}\n"
    ),
    "func_attribute": BASE + (
        "void i(const void* k) {\n"
        "  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 1 << 17);\n"
        "}\n"
    ),
    "pragma_unroll": BASE + (
        "__device__ void j(float* x) {\n"
        "#pragma unroll\n"
        "  for (int i = 0; i < 4; ++i) x[i] = 0.f;\n"
        "}\n"
    ),
}


def test_hostile_sources_rejected() -> None:
    for name, source in HOSTILE.items():
        errors = admission.scan_cuda_source(source, f"{name}.cu")
        assert errors, f"hostile fixture {name} was admitted"


def test_benign_sources_admitted() -> None:
    for name, source in BENIGN.items():
        errors = admission.scan_cuda_source(source, f"{name}.cu")
        assert not errors, f"benign fixture {name} rejected: {errors[:3]}"


def test_structural_admission_accepts_oracle(tmp_path) -> None:
    root = tmp_path / "lab"
    shutil.copytree(TASK_ROOT / "oracle", root)
    report = admission.validate_submission(root)
    assert report["ok"], report["errors"]
    assert report["final_version"] == 1


def test_structural_admission_rejects_extra_files(tmp_path) -> None:
    root = tmp_path / "lab"
    shutil.copytree(TASK_ROOT / "oracle", root)
    (root / "candidates/kernel_v001/helper.cuh").write_text("// extra\n")
    report = admission.validate_submission(root)
    assert not report["ok"]
    assert any("unexpected candidate entry" in error for error in report["errors"])


def test_structural_admission_rejects_symlink(tmp_path) -> None:
    root = tmp_path / "lab"
    shutil.copytree(TASK_ROOT / "oracle", root)
    # kernel_v002 is a real (unselected) oracle candidate; use a free version slot.
    target = root / "candidates/kernel_v099"
    target.mkdir()
    (target / "submission.cu").symlink_to(
        root / "candidates/kernel_v001/submission.cu"
    )
    report = admission.validate_submission(root)
    assert not report["ok"]


def test_final_version_must_select_existing(tmp_path) -> None:
    root = tmp_path / "lab"
    shutil.copytree(TASK_ROOT / "oracle", root)
    (root / "FINAL_VERSION").write_text("7\n")
    report = admission.validate_submission(root)
    assert not report["ok"]
    assert any("missing kernel_v007" in error for error in report["errors"])
