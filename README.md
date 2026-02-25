# BarraCUDA

An open-source CUDA C++ compiler written from scratch in C99 that takes `.cu` files and compiles them to AMD GPU machine code, with more architectures planned. No LLVM, no dependencies, and no permission asked.

This is what happens when you look at NVIDIA's walled garden and think "how hard can it be?" The answer is: quite hard, actually, but I did it anyway.

**UPDATE: RDNA 2, 3 and 4 now supported.** `--gfx1030` / `--gfx1100` / `--gfx1200`

## What It Does

Takes CUDA C source code, the same `.cu` files you'd feed to `nvcc`, and compiles them to AMD RDNA 2 (gfx1030), RDNA 3 (gfx1100) and RDNA 4 (gfx1200) binaries. No LLVM. No HIP translation layer. No "convert your CUDA to something else first." Just a lexer, a parser, an IR, and a hand-written instruction selection backend that would make a compiler textbook weep.

```
┌──────────────────────────────────────────────────────────────┐
│                     BarraCUDA Pipeline                        │
├──────────────────────────────────────────────────────────────┤
│  Source (.cu)                                                │
│       ↓                                                      │
│  Preprocessor → #include, #define, macros, conditionals      │
│       ↓                                                      │
│  Lexer → Tokens                                              │
│       ↓                                                      │
│  Parser (Recursive Descent) → AST                            │
│       ↓                                                      │
│  Semantic Analysis → Type checking, scope resolution         │
│       ↓                                                      │
│  BIR (BarraCUDA IR) → SSA form, typed instructions           │
│       ↓                                                      │
│  mem2reg → Promotes allocas to SSA registers                  │
│       ↓                                                      │
│  Instruction Selection → AMDGPU machine instructions         │
│       ↓                                                      │
│  Register Allocation → VGPR/SGPR assignment                  │
│       ↓                                                      │
│  Binary Encoding → GFX10/GFX11/GFX12 instruction words         │
│       ↓                                                      │
│  ELF Emission → .hsaco ready for the GPU                     │
│       ↓                                                      │
│  Your kernel runs on silicon that NVIDIA doesn't control     │
└──────────────────────────────────────────────────────────────┘
```

Every single encoding has been validated against `llvm-objdump` with zero decode failures. I didn't use LLVM to compile, but I did use it to check my homework.

## Building

```bash
# It's C99. It builds with gcc. There are no dependencies.
make

# That's it. No cmake. No autoconf. No 47-step build process.
# If this doesn't work, your gcc is broken, not the Makefile.
```

### Requirements

- A C99 compiler (gcc, clang, whatever you've got)
- A will to live (optional but recommended)
- LLVM is NOT required. BarraCUDA does its own instruction encoding like an adult.

## Usage

```bash
# Compile to AMD GPU binary (RDNA 3, default)
./barracuda --amdgpu-bin kernel.cu -o kernel.hsaco

# Compile for RDNA 2
./barracuda --amdgpu-bin --gfx1030 kernel.cu -o kernel.hsaco

# Compile for RDNA 4
./barracuda --amdgpu-bin --gfx1200 kernel.cu -o kernel.hsaco

# Dump the IR (for debugging or curiosity)
./barracuda --ir kernel.cu

# Just parse and dump the AST
./barracuda --ast kernel.cu

# Run semantic analysis
./barracuda --sema kernel.cu
```

## Runtime Launcher

BarraCUDA includes a minimal HSA runtime (`src/runtime/`) for dispatching compiled kernels on real AMD hardware. Zero compile-time dependency on ROCm — loads `libhsa-runtime64.so` at runtime via `dlopen`.

```bash
# Compile the runtime and example together
gcc -std=c99 -O2 -I src/runtime \
    examples/launch_saxpy.c src/runtime/bc_runtime.c \
    -ldl -lm -o launch_saxpy

# Compile a kernel and run it
./barracuda --amdgpu-bin -o test.hsaco tests/canonical.cu
./launch_saxpy test.hsaco
```

Requires Linux with ROCm installed. See `examples/launch_saxpy.c` for a complete example. **Not yet tested on real hardware** — if you have an AMD GPU, we'd love a test report ([#39](https://github.com/Zaneham/BarraCUDA/issues/39)).

## What Works

 The following CUDA features compile to working GFX10/GFX11/GFX12 machine code:

### Core Language
- `__global__`, `__device__`, `__host__` function qualifiers
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` builtins
- Structs, enums, typedefs, namespaces
- Pointers, arrays, pointer arithmetic
- All C control flow: `if`/`else`, `for`, `while`, `do-while`, `switch`/`case`, `goto`/`label`
- Short-circuit `&&` and `||`
- Ternary operator
- Templates (basic instantiation)
- Multiple return paths, `continue`, `break`

### CUDA Features
- `__shared__` memory (allocated from LDS, properly tracked)
- `__syncthreads()` → `s_barrier`
- Atomic operations: `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`
- Warp intrinsics: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
- Warp votes: `__ballot_sync`, `__any_sync`, `__all_sync`
- Vector types: `float2`, `float3`, `float4`, `int2`, `int3`, `int4` with `.x`/`.y`/`.z`/`.w` access
- Half precision: `__half`, `__float2half()`, `__half2float()`
- `__launch_bounds__` (parsed, propagated, enforces VGPR caps)
- Cooperative groups: `cooperative_groups::this_thread_block()` with `.sync()`, `.thread_rank()`, `.size()`
- Operator overloading
- Math builtins: `sqrtf`, `rsqrtf`, `expf`, `exp2f`, `logf`, `log2f`, `log10f`, `sinf`, `cosf`, `tanf`, `tanhf`, `powf`, `fabsf`, `floorf`, `ceilf`, `truncf`, `roundf`, `rintf`, `fmaxf`, `fminf`, `fmodf`, `copysignf`
- `__constant__` memory, `__device__` globals

### Compiler Features
- Full C preprocessor: `#include`, `#define`/`#undef`, function-like macros, `#ifdef`/`#ifndef`/`#if`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`, `-I`/`-D` flags
- Error recovery (reports multiple errors without hanging)
- Source location tracking in IR dumps
- Struct pass-by-value

## Example

```cuda
__global__ void vector_add(float *c, float *a, float *b, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
```

```
$ ./barracuda --amdgpu-bin vector_add.cu -o vector_add.hsaco
wrote vector_add.hsaco (528 bytes code, 1 kernels)
```

No LLVM required :-) 


## What Doesn't Work (Yet)

Being honest about limitations is important. Here's what's missing:

- Parameter reassignment in `__device__` functions (use local variables)
- Textures and surfaces
- Dynamic parallelism (device-side kernel launch)
- Multiple translation units
- Host code generation (only device code is compiled)

None of these are architectural blockers. They're all "haven't got round to it yet" items.

## Test Suite

14 test files, 35+ kernels, ~1,700 BIR instructions, ~27,000 bytes of machine code:

- `vector_add.cu` - The "hello world" of GPU computing
- `cuda_features.cu` - Atomics, warp ops, barriers, gotos, switch, short-circuit
- `test_tier12.cu` - Vectors, shared memory, operator overloading
- `notgpt.cu` - AI-generated CUDA with extremely sarcastic comments (tiled SGEMM, reductions, histograms, prefix scan, stencils, half precision, cooperative groups, and the "kitchen sink" kernel)
- `stress.cu` - N-body simulation, nested control flow, bit manipulation, struct pass-by-value, chained function calls
- `canonical.cu` - Canonical patterns from NVIDIA samples adapted for the parser
- `test_errors.cu` - Deliberate syntax errors to verify error recovery
- `test_launch_bounds.cu` - `__launch_bounds__` parsing and VGPR cap enforcement
- `test_coop_groups.cu` - Cooperative groups lowering
- `mymathhomework.cu` - Trig identities, exponential growth, Newton-Raphson, log laws, hyperbolic functions, floor/ceil/round, power rule, clamping
- Plus preprocessor tests, template tests, unsigned integer tests

## Roadmap

### Near Term: Hardening

Fix the known gaps: integer literal suffixes, `const`, parameter reassignment. These are all small parser/lowerer changes. The goal is to compile real-world `.cu` files without modifications.

### Medium Term: Optimisation

The generated code works but isn't winning any benchmarks. Priorities:

- Instruction scheduling (hide memory latency)
- Better register allocation (currently linear scan, consider graph colouring)
- Constant folding and dead code elimination
- Loop-invariant code motion
- Occupancy tuning based on register pressure

### Long Term: More Architectures

The IR (BIR) is target-independent. The backend is cleanly separated. Adding a new target means writing a new `isel` + `emit` pair. Candidates:

- **Tenstorrent** - RISC-V based AI accelerators. Open ISA. Very different execution model (tile-based, not SIMT) but the IR maps well.
- **Intel Arc** - Xe architecture. Would give BarraCUDA coverage across all three major GPU vendors.
- **RISC-V Vector Extension** - For when GPUs are too mainstream and you want to run CUDA on a softcore.

## Why Does This Exist?

1. Because CUDA shouldn't require an NVIDIA GPU
2. Because HIP translation is a workaround, not a solution
3. Because a few thousand lines of C is a more honest compiler than a million lines of LLVM
4. Because someone should prove it's possible to write a GPU compiler backend by hand
5. Because open hardware deserves open compilers
6. I wasn't going to play video games anyway (lie)

## GFX10/GFX11/GFX12 Encoding Notes (For The Brave)

If you're considering writing your own AMDGPU backend, here are the things that will ruin your afternoon:

- SOP1 prefix is `0xBE800000`, not what you'd expect from the docs
- SOPC prefix is `0xBF000000`
- VOP3 VDST is at bits `[7:0]`, not `[15:8]` like a sensible person would assume
- Null SADDR is `0x7C` for global memory, `0xFC` for scratch
- RDNA 3 is Wave32 by default, not Wave64 like GCN
- The ISA manual is 500 pages and contradicts itself at least twice
- GFX12 FLAT/GLOBAL OP field is at `[21:14]`, not `[20:13]` like the RDNA4 PDF claims. Trust the machine-readable ISA, not the PDF
- GFX10 SMEM prefix is `0x3C` (not `0x3D`), with different bit layout for SDATA/SBASE/OP
- GFX10 VOP3 prefix is `0x34` (not `0x35`)
- GFX10 FLAT/GLOBAL DW1 has DATA and SADDR swapped vs GFX11, null SADDR is `0x7D`
- GFX10 waitcnt SIMM16 layout: `[15:14]=vmcnt[5:4] [13:11]=expcnt [9:4]=lgkmcnt [3:0]=vmcnt[3:0]`
- Nearly every hw_opcode is renumbered between GFX10 and GFX11

`amdgpu_emit.c` is a testament to reading those pages so you don't have to.

## Contact

Found a bug? Want to discuss the finer points of AMDGPU instruction encoding? Need someone to commiserate with about the state of GPU computing?

**zanehambly@gmail.com**

Open an issue if theres anything you want to discuss. Or don't. I'm not your mum.

Based in New Zealand, where it's already tomorrow and the GPUs are just as confused as everywhere else.

## License

Apache 2.0. Do whatever you want. If this compiler somehow ends up in production, I'd love to hear about it, mostly so I can update my LinkedIn with something more interesting than wrote a CUDA compiler for fun.

## Acknowledgements

- **Steven Muchnick** for *Advanced Compiler Design and Implementation*. If this compiler does anything right, that book is why.
- **Low Level** for the Zero to Hero C course and the YouTube channel. That's where I learnt C.
- **Abe Kornelis** for being an amazing teacher. His work on the [z390 Portable Mainframe Assembler](https://github.com/z390development/z390) project is well worth your time.
- To the people who've sent messages of kindness and critique, thank you from a forever student and a happy hobbyist.
- My Granny, Grandad, Nana and Baka. Love you x

*He aha te mea nui o te ao. He tāngata, he tāngata, he tāngata.*

What is the most important thing in the world? It is people, it is people, it is people.

---

