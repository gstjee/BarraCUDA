# BarraCUDA

An open-source CUDA compiler that targets AMD GPUs, with more architectures planned. Written in 15,000 lines of C99. Zero LLVM dependency. Compiles `.cu` files straight to GFX11 machine code and spits out ELF `.hsaco` binaries that AMD GPUs can actually run.

This is what happens when you look at NVIDIA's walled garden and think "how hard can it be?" The answer is: quite hard, actually, but I did it anyway.

## What It Does

Takes CUDA C source code, the same `.cu` files you'd feed to `nvcc`, and compiles them to AMD RDNA 3 (gfx1100) binaries. No LLVM. No HIP translation layer. No "convert your CUDA to something else first." Just a lexer, a parser, an IR, and roughly 1,700 lines of hand-written instruction selection that would make a compiler textbook weep.

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
│  Binary Encoding → GFX11 instruction words                   │
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
# Compile to AMD GPU binary
./barracuda --amdgpu-bin kernel.cu -o kernel.hsaco

# Dump the IR (for debugging or curiosity)
./barracuda --ir kernel.cu

# Just parse and dump the AST
./barracuda --ast kernel.cu

# Run semantic analysis
./barracuda --sema kernel.cu
```

## What Works

 The following CUDA features compile to working GFX11 machine code:

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

## Architecture

| File | Lines | What It Does |
|------|-------|-------------|
| `lexer.c` | 747 | Tokenises CUDA C source |
| `preproc.c` | 1,370 | C preprocessor (macros, includes, conditionals) |
| `parser.c` | 1,500 | Recursive descent parser → AST |
| `sema.c` | 1,725 | Type checking, scope resolution, overload resolution |
| `bir.c` + `bir_lower.c` | 3,032 | SSA intermediate representation + AST→BIR lowering |
| `bir_mem2reg.c` | 965 | Promotes stack allocas to SSA registers |
| `bir_print.c` | 579 | IR pretty printer with source location annotations |
| `amdgpu_isel.c` | 1,788 | Instruction selection: BIR → AMDGPU machine ops |
| `amdgpu_emit.c` | 1,735 | Register allocation + GFX11 binary encoding + ELF emission |
| `main.c` | 317 | CLI driver |
| **Total** | **15,117** | |

All data structures use pre-allocated fixed-size arrays. No malloc in hot paths. No recursion. Bounded loops everywhere. The kind of code that would make JPL's coding standards committee nod approvingly before going back to landing things on Mars.

## What Doesn't Work (Yet)

Being honest about limitations is important. Here's what's missing:

- 2D array declarations in shared memory (`__shared__ float a[16][16]`, flatten to 1D)
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
3. Because 15,000 lines of C is a more honest compiler than a million lines of LLVM
4. Because someone should prove it's possible to write a GPU compiler backend by hand
5. Because open hardware deserves open compilers
6. I wasn't going to play video games anyway (lie)

## GFX11 Encoding Notes (For The Brave)

If you're considering writing your own AMDGPU backend, here are the things that will ruin your afternoon:

- SOP1 prefix is `0xBE800000`, not what you'd expect from the docs
- SOPC prefix is `0xBF000000`
- VOP3 VDST is at bits `[7:0]`, not `[15:8]` like a sensible person would assume
- Null SADDR is `0x7C` for global memory, `0xFC` for scratch
- RDNA 3 is Wave32 by default, not Wave64 like GCN
- The ISA manual is 500 pages and contradicts itself at least twice

All 1,735 lines of `amdgpu_emit.c` are a testament to reading those pages so you don't have to.

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

