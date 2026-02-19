# Contributing to BarraCUDA

You're more than welcome to submit a PR. I'm happy to look at it and give it a fair shake.

That said, this project has a particular style and a particular set of rules. They're not arbitrary because I just wanna, they exist because they eliminate entire categories of bugs. If your PR follows them, we'll get along swimmingly. 

If you're reading this and going "Ah heck I don't think I can do that, I'll get it wrong". Thats perfectly fine. Again, you're more than welcome to submit a pr and I am always happy to guide and assist. I am always learning myself. 

## Coding Standards

BarraCUDA is written in a defensive C99 style designed for reliability. The rules are simple:

- **No dynamic allocation in hot paths.** Pre-allocated, fixed-size buffers. If a pool overflows, return a sentinel — never corrupt a counter.
- **No recursion.** Every function call is a known-depth call. Stack usage is predictable.
- **All loops must be bounded.** If you're iterating to a fixpoint, there's a guard counter. No infinite loops, no "this should always converge."
- **Bounds-check array accesses** from external or untrusted indices. Trust internal bookkeeping, verify everything else.
- **Stack-allocated where possible.** Deterministic behaviour, deterministic cleanup.
- **Strict error checking.** Check return values. Handle the failure path.

The rationale is straightforward: if you can't have unbounded allocation, you can't have memory leaks. If you can't have recursion, you can't have stack overflows. If every loop is bounded, you can't hang. This style eliminates a large set of bugs by construction rather than by testing.

### Comments

Ideally minimal. 

Yes I know some of my comments in there have some pearlers but that is okay! 

Section headers look like this:
```c
/* ---- Section Name ---- */
```

Humour is welcome. Arrogance is not.

## What's Useful

Check `Issues` for current priorities. At time of writing, the most impactful areas are:

- **Language features:** bare `unsigned`, literal suffixes, `const`, parameter reassignment
- **Backend work:** New architecture targets (see below)
- **Test cases:** Real CUDA kernels that break things are genuinely valuable

## What's Less Useful

- Cosmetic README rewrites
- Drive-by reformatting or linter passes
- Adding abstraction layers, feature flags, or "improvements" that weren't asked for
- Wrapping working code in error handling for scenarios that can't happen

If in doubt, open an issue first. A two-minute conversation saves everyone time.

## GPU Architectures

BarraCUDA currently targets AMD RDNA 3 (gfx1100). That is by no means the only architecture on the roadmap.

The compiler architecture is purposefully backend-agnostic: the frontend lowers to BIR (BarraCUDA IR) in SSA form, and each backend is a self-contained target that consumes BIR. Adding a new GPU architecture means writing a new instruction selector and emitter — the rest of the pipeline is shared.

If you're a deep tech startup and you need CUDA support for your hardware, you're encouraged to reach out. I'm actively interested in expanding target support.

## Building & Testing

```bash
# Build (requires LLVM 18 for disassembly verification only — the compiler itself is zero-LLVM)
make

# Run the emulator test suite
python tests/emu/run_emu.py
```

Verify your changes don't introduce encoding regressions:
```bash
llvm-objdump -d --mcpu=gfx1100 output.hsaco
# Zero decode failures = good/bueno!
```

## License

BarraCUDA is Apache 2.0. By submitting a PR, you agree your contribution is licensed under the same terms.
