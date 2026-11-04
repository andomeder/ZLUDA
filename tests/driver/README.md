# Driver-layer diagnostic tests (CUDA driver API against ZLUDA libnvcuda.so)

These are standalone C programs + PTX kernels that exercise the ZLUDA CUDA
driver layer (`libnvcuda.so`) on real hardware. No CUDA toolkit is required:
compile against the headers shipped with the `nvidia/cuda_runtime` python
package and link against `target/release/libnvcuda.so`.

## Build and run

```sh
CUDA_INC=$HOME/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/nvidia/cuda_runtime/include
ZLUDA_LIB=/path/to/intel-zluda/ZLUDA/target/release

gcc -o vector_add vector_add.c -I"$CUDA_INC" -L"$ZLUDA_LIB" -lnvcuda
LD_LIBRARY_PATH="$ZLUDA_LIB" ./vector_add
```

## Verified working on the Arc B580 (2026-08-11)

- `vector_add.c` / `vectorAdd_11.ptx`: full driver stack (cuInit ->
  cuDeviceGet -> cuCtxCreate -> cuModuleLoadData -> cuMemAlloc/cuMemcpy ->
  cuLaunchKernel -> cuMemcpyDtoH). Correct for 1024 and 1M elements.
- `stress.c`: 1M-element vector add, verified against CPU.
- `grid2d.c` / `vectorAdd2d_11.ptx`: 2D grid (16x16) vector add, uses
  %ctaid.x, %ctaid.y, %nctaid.x, %ntid.x, %tid.x and mad.lo.s32 with mixed
  operands. 65536 elements correct.
- `ctaid2d.c`/`ctaid2d.ptx`: per-block (ctaid.x, ctaid.y) dump for arbitrary
  2D grids. Confirms WorkgroupId.x/y are correct on Battlemage.
- `ctaid3.c`, `ctaid4.c`, `ctaid_large.c`, `ctaidrace.c`, `ctaidy.c`,
  `tid_raw.c`, `tid_test.c`: 1D/2D builtin checks (tid, ctaid x/y).
  Note: tests where all blocks write one slot are races (last writer wins)
  and only verify that non-zero values flow.

## Fixed: mad.lo.s32 with a special-register operand was dropped

`mad.lo.s32` (and any Arg4/Arg3 arithmetic) with a vector-member special
register operand (`%tid.x`, `%ctaid.x`, `%ntid.x` used directly) failed to
parse: the grammar's `Operand` rule did not accept `VecMember` operands, so
lalrpop error-recovered and silently dropped the whole instruction. The
result register was never written and consumers read uninitialized garbage.

Fix in `ptx/src/ptx.lalrpop`: added a `MemberOperand` (VecMember) alternative
to the `Operand` rule. `ctaid5`/`ctaid6` style kernels now compile and run
correctly. vectorAdd never hit this because its mad used only plain registers.

## Pure Level Zero probes (no ZLUDA)

- `ze_wgid1d.c`, `ze_wgid2d.c`: compile the SPIR-V that ZLUDA generates
  directly via zeModuleCreate + zeCommandListAppendLaunchKernel, isolating
  whether behavior comes from ZLUDA or IGC/L0. Build with `-lze_loader`.
  The SPIR-V binaries are produced by the ptx crate test harness.

## Other

- `ze_smoke.c`: plain Level Zero smoke test (zeInit/zeDriverGet/zeDeviceGet).

## Driver API surface expansion (2026-08-15)

Real apps probe the driver on startup; these were stubs returning
`CUDA_ERROR_NOT_SUPPORTED` (or silently garbage). Now implemented in
`zluda/src/impl/`:

- `cuDeviceGetAttribute`: expanded from ~15 to ~40+ attributes, mapping L0
  device/compute properties. Unknown attributes now return defined 0 instead
  of leaving `*pi` uninitialized.
- `cuDeviceComputeCapability` + `cuDeviceGetProperties` (sm_80, L0-derived
  dims/limits).
- `cuFuncGetAttribute`: now answers MAX_THREADS_PER_BLOCK, SHARED_SIZE_BYTES,
  LOCAL_SIZE_BYTES, PTX_VERSION, BINARY_VERSION, etc.
- `cuFuncSetAttribute`: accepts MAX_DYNAMIC_SHARED_SIZE_BYTES and
  PREFERRED_SHARED_MEMORY_CARVEOUT (dynamic shared size is passed per-launch
  via the sharedMemBytes arg, so no per-function state is needed).
- `cuModuleLoadFatBinary`: new `zluda/src/impl/fatbin.rs` parses CUDA fatbin
  containers (both the `0x466243B1` fatbinc wrapper and the `0xBA55ED50`
  plain frame), picks the best PTX record (64-bit, uncompressed or LZ4), and
  feeds it to the SPIR-V translator. ELF/cubin records are skipped.

### Verified on B580

- `atomic_shared_test.c`: global atomics (256 threads -> 256), dynamic/static
  shared memory with barrier + thread-0 readback (out=127), shared-memory
  atomics (256 threads -> 256), and cross-block atomics (8 blocks x 32 ->
  256). All correct.
- `fatbin_test.c`: both fatbin container formats load via
  cuModuleLoadFatBinary and run the atom kernel correctly (256).
- `ptx_load_check.c`: helper that reports LOAD OK/FAIL for a single PTX.
- `mkfatbin.py`: packs a PTX into .fatbin/.fatbinc test containers.

Note: `.extern .shared .b32 foo [];` (dynamic shared) needs a space before
`[]` to parse, and the translator's dynamic-shared rewrite only resolves
symbol-referenced shared operands (raw register addressing fails with
UnknownSymbol). The tests use static `.shared` arrays instead.

## Events + host-pinned memory (2026-08-15)

Real apps use events for timing/sync and pinned host memory for fast
transfers; these were all stubs. Now implemented:

- `cuEventCreate`/`Record`/`Query`/`Synchronize`/`Destroy_v2`/`ElapsedTime`
  in `zluda/src/impl/event.rs`: events wrap L0 host-visible events from the
  device pool; record appends a barrier to the stream that signals the
  event; elapsed time uses host wall-clock at record.
- Host pinned memory in `zluda/src/impl/memory.rs`: `cuMemAllocHost_v2`,
  `cuMemFreeHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer_v2`,
  `cuMemHostGetFlags` via L0 `zeMemAllocHost` (device-accessible host mem).
- `cuStreamQuery` (busy/ready via last event) + `cuMemGetInfo_v2`.
- L0 crate gained `mem_alloc_host`.

### Verified on B580

- `event_pinned_test.c`: mem info, vector add on host-pinned pointers
  (cuMemAllocHost + cuMemHostGetDevicePointer, no cuMemAlloc/cuMemcpy),
  event create/record/query/synchronize/elapsed, and stream query
  busy->ready. All correct.

Deadlock note: the first event implementation held the global state lock
while taking the stream lock, which deadlocked (std Mutex is not
reentrant). Fixed by checking liveness under the lock, then recording
outside it.

## Pointer attributes, async memcpy, stream APIs, 2D/3D memcpy (2026-08-15)

- `pointer_attr_test.c`: cuPointerGetAttribute / cuPointerGetAttributes on
  device + host-pinned allocations (MEMORY_TYPE host=1/device=2, ranges,
  ordinal, interior-pointer lookup), unknown pointer -> INVALID_VALUE.
- `async_memcpy_test.c`: cuMemcpyHtoDAsync/DtoDAsync/DtoHAsync on a stream
  ordered by events, sync cuMemcpyDtoD, legacy-stream cuMemcpyAsync.
- `stream_api_test.c`: cuStreamGetFlags (incl. CU_STREAM_NON_BLOCKING
  round-trip), cuStreamGetPriority, cuStreamWaitEvent cross-stream ordering,
  cuStreamAddCallback host callback.
- `memcpy2d3d_test.c`: cuMemcpy2D host->device with 512-byte pitch and back,
  cuMemcpy3D depth=2.

Notes:

- cuPointerGetAttribute relies on a per-context PointerRegistry populated by
  cuMemAlloc/cuMemAllocHost/cuMemHostAlloc and cleared on free.
- Sync DtoD must synchronize the legacy stream (the enqueue itself is async).
- Intel's L0 driver corrupts zeCommandListAppendMemoryCopyRegion output when
  dstPitch > width, so 2D/3D copies are decomposed into per-row 1D copies.

## A fatbin that carries only SASS (2026-11-04)

A fatbin built with `-gencode arch=compute_75,code=sm_75` holds NVIDIA SASS
(kind=2 ELF records) and no PTX, and the translator reads PTX, so the loader
refuses such an image with `CUDA_ERROR_NOT_SUPPORTED`. `zluda/src/impl/sass.rs`
adds a way through it: the cubin is lifted back to PTX
(cubin -> CuLifter -> LLVM IR -> `llc` -> PTX) and continues through the
existing PTX -> SPIR-V -> IGC path. It is **off by default** (the lifter has no
licence) and enabled per process:

    ZLUDA_SASS_LIFT=1
    ZLUDA_SASS_LIFTER=/tmp/opencode/sasslifter/CuLifter   # the checkout
    ZLUDA_SASS_CUOBJDUMP=<cuda toolkit>/bin/cuobjdump     # or on PATH
    ZLUDA_SASS_CACHE=~/.cache/zluda/sass                  # the default
    ZLUDA_SASS_PYTHON=<venv>/bin/python                   # default: the venv
                                                          # next to the checkout
    ZLUDA_SASS_LLC=llc                                    # default: PATH

A lifted kernel is cached per cubin (`<cache>/<sha256 of the cubin>/`), so a
second load of the same image is a file read. An image that carries any PTX
record is still served by its PTX and nothing is lifted.

`sass_lift_test.c` is the hardware check. It loads a container whose only
record is the real `torch.mul` elementwise kernel out of
`libtorch_cuda.so` (the 4-wide `vectorized_elementwise_kernel` whose second
argument is a by-value `std::array<char*, 3>`) through both
`cuLibraryLoadData` + `cuLibraryGetKernel` and `cuModuleLoadFatBinary` +
`cuModuleGetFunction`, launches it with the parameter block the kernel reads
from the constant bank, and compares 4096 and 3000 (tail path) elements
bit-exactly against the CPU.

```sh
CUDA_INC=.../nvidia/cuda_runtime/include
ZLUDA_LIB=.../ZLUDA/target/release
gcc -O2 -o sass_lift_test sass_lift_test.c -I"$CUDA_INC" -L"$ZLUDA_LIB" -lnvcuda

# a container with one ELF record and no PTX, from a cubin
python3 mkfatbin.py path/to/kernel.cubin sass_mul --elf

LD_LIBRARY_PATH="$ZLUDA_LIB" ./sass_lift_test sass_mul.fatbin
    -> cuLibraryLoadData: CUDA_ERROR_NOT_SUPPORTED
       REFUSED WITHOUT THE LIFT   (the behaviour without the variable)

ZLUDA_SASS_LIFT=1 ZLUDA_SASS_LIFTER=<checkout> ZLUDA_SASS_CUOBJDUMP=<cuobjdump> \
LD_LIBRARY_PATH="$ZLUDA_LIB" ./sass_lift_test sass_mul.fatbin
    -> cuLibraryLoadData: OK
       cuLibraryGetKernel: OK (497 ms, includes lift + SPIR-V + IGC)
       N=4096 grid=8: wrong 0/4096
       N=3000 (tail path): wrong 0/3000
       SASS FATBIN TEST PASSED
```

Measured on the B580: the first load of that kernel costs 475 ms in the lift
(0.2 s disassembly, 0.3 s CuLifter, 0.05 s `llc`) plus 20 ms of translation and
IGC; a later load is a 0.03 ms cache read plus the same 20 ms of IGC. A kernel
the lifter cannot handle fails the load of that kernel with the stage that
failed (lift, llc, translate, IGC) and the kernel's name; the module is never
built with the kernel silently missing.

A cubin whose header names no architecture the lifter implements fails with
`the cubin does not name a known architecture (e_flags=...)`, and one whose
architecture the lifter does not implement (`sm_90`, everything above
`sm_86`) with `sm_90 is not a liftable architecture`.

Regenerate the cubin from the library with NVIDIA's own tools:

```sh
cuobjdump -lelf libtorch_cuda.so | grep sm_75          # the ELF images
cuobjdump -xelf libtorch_cuda.313.sm_75.cubin libtorch_cuda.so
```
