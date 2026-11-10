# CONCLUSION — `apu-expert-pool` is CLOSED (2026-11-10).  Do not re-enter this path.

**Superseded by [`wip/uma-static-window/`](../../../wip/uma-static-window/README.md).**  This directory is kept
as the *evidence base*, not as a plan: it holds the eleven probes and the measurements that **disproved** the
campaign's own thesis.  Read this page before any of the others; the campaign's `README.md`, `IMPLEMENTATION.md`
and `HANDOVER.md` describe a design that the numbers below killed, and the last of them
(`NEXT-SESSION-PROMPT.md`) is obsolete.

> **CORRECTION (2026-11-10, added by `wip/uma-static-window` P0 — the record is otherwise left as written).**  The
> three 122B figures below (`10.58 / 10.90 / 10.89` t/s and the arena triples) have **no arm behind them**: this
> directory's `MEASUREMENTS.md` contains no 122B section, `README.md` §11 says "the **>RAM acceptance arm has not
> been run**", and `HANDOVER.md` opens with "the case the campaign exists for — a model LARGER than RAM — has not
> been run".  The first *recorded* 122B arms are in `wip/uma-static-window/p0/RESULTS.md` §C: `-ncmoe 48 -c 8192`,
> cache off, marginal **8.0 t/s** over tokens 64..320 and **5.7 t/s** over 320..600 (the rate falls with length —
> the set cannot stay in the page cache), and `-ncmoe 24` (57.4 GiB of experts on the device) **8.5** — i.e. no
> placement win at all, because on UMA the device pool and the page cache are one resource.  The *qualitative*
> conclusion below (do not port the discrete tier to UMA; the medium is the same on both sides) stands on the
> probes, which are recorded.

## What the campaign believed

That the host-expert / expert-arena tier shipped for discrete cards could be opened on a UMA APU (gfx1150/1151
and newer) by replacing the blanket refusal with a RAM budget: keep the arena, size it from `MemAvailable`
instead of "free VRAM", and let an eviction move no data because the arena is host-writable.

## What is actually true (measured, `halo`, gfx1151, 123 GiB, ROCm 10.1.0)

1. **The slab's arena is NOT host-writable.**  `probe/11-fill-path.cpp`: a `hipMemCreate(Pinned)` + reserve +
   map range faults on a host dereference (`hipMalloc` does not — the campaign's early probes measured
   `hipMalloc` and generalised).  So every arena fill is a **copy**, and "an eviction moves no data" is only
   true of the table entry, not of the bytes.
2. **The arena buys nothing on UMA.**  A registered-anon buffer reads at **198.7 GB/s**, the arena at
   **194.3 GB/s** — the same DRAM either side.  The tier's copy is pure overhead here; on a discrete card the
   same copy buys ~13x (VRAM ~640 GB/s against a 41.1–55.8 GB/s link), which is the whole reason the discrete
   design exists.
3. **At the size class the campaign was built for, the tier loses.**  Qwen3.5-122B-A10B Q8_0 (123.4 GiB, of
   which **114.8 GiB is expert tensors**), `-ncmoe 48 -c 32768`:
   | config | decode t/s (3 x 600-token passes) |
   |---|---|
   | cache off — the CPU expert path | **10.58 / 10.90 / 10.89** |
   | arena 32 GiB | 3.75 / 3.82 / 3.04 |
   | arena auto (~70 GiB) | 3.29 / 2.84 / 2.36 |
   Every arena arm is 3x slower, with no memory distress (75–110 GiB `MemAvailable`).
4. **The slab's own floor decides whether the tier exists at all.**  For that model the slab needs work
   2.7 GiB + a 2 GiB arena floor + a 4.1 GiB ring ≈ **9 GiB** before it can be created; below that the cache
   prints `STREAMING the experts from the host (stock -ncmoe behaviour)` and the run is the CPU path.  So "a
   small arena plus a big pool" is unreachable: 2 GiB and 8 GiB arms were never tier arms.
5. **Zero arena is not a supported state.**  With `MOE_UMA_POOL_ONLY=1` no cache table registers at all
   (`table_pool_locked` never runs, the teardown census never prints because `g_tables` is empty) and every
   MoE op falls back to the CPU: measured 772 % CPU (48 % of 16 cores), GPU 27–34 %, identical to the
   untiered number (10.9 t/s).
6. **A read-only-registered pool cannot be filled at speed.**  Registering first and then filling gives
   **70–210 MB/s** (page-fault bound) against **~7 GB/s** into the writable pool — the fill must precede the
   registration, which an evicting cache cannot promise for the whole tier.  Per-slot register-after-fill
   (~15 µs per 3.3 MB slot) is the only viable form.
7. **Where the tier does win:** when the routed set fits the arena.  Qwen3.8-Flash-Next IQ4_NL (A3B) under a
   31 GiB ballast, `-ncmoe 24`: arena 21 GiB = **37.4 / 43.6 / 44.0 t/s** against the CPU path's
   29.7 / 30.8 / 31.0 (**+42 %**), MTP drafter alive (draft acceptance ~0.53).

## Why the numbers look like this (the rule the campaign did not have)

At batch 1 a sparse MoE decode is **memory-movement bound**: the 122B routes ~3.8 GiB of expert bytes per
token.  The CPU reads them in place from the page cache at ~41 GB/s effective (10.9 t/s).  A GPU reading a
resident buffer could do the same bytes at ~194 GB/s (~51 t/s) — but the tier adds a **copy per miss** and a
**per-op host round-trip** (the remap readback, 144 routed ops/token), and on UMA neither buys a faster
medium.  ~85 % of the wall clock goes to the tier's machinery, not to the reads.  A discrete card wins because
the same machinery relocates the bytes into memory that is 13x faster; an APU has no such memory.

Corollary, and the reason the new campaign exists: on UMA the goal is **not residency** (the page cache already
holds the bytes) but **the GPU reading them where they are, with no copy and no per-op round-trip**.

## What this campaign did leave behind (kept, not discarded)

Four defects were found and fixed in the fork (branch `wip/apu-expert-pool`, commit `70e0eda4a`); all four are
real and affect the **shipped** path, not just UMA:

- the UMA arena allowance is now **net of what is already on the device** (`cap − device_used`) and the slab
  maps it **on demand** (`ggml_cuda_slab_arena_alloc` grows it), instead of mapping "free VRAM − reserve"
  (measured: 84.00 GiB mapped to hold 22.8 GiB of tables → a 51 GiB swap storm that never finished the load);
- `pool_clamp_slots_to_ram()` keys on `pool_ram_pct_eff()`, not the raw `-1` "auto" default, so the pool's RAM
  ceiling actually binds (measured: a 22 775 MiB mirror against a 12 100 MiB ceiling);
- `MOE_UMA_VRAM_PCT` is parsed once and travels with the free-memory snapshot, so the advisory and the arena
  budget are the same arithmetic (the advisory used to hardcode 75 %);
- `--lazy-mode auto` resolves against the **declared load mode**, not the device's advertised
  `caps.mmap_support`, so an APU with an explicit `-lm mmap` no longer loads a 27 465 MiB table resident.

Also kept: `probe/11-fill-path.cpp` (the fill-path matrix: `hipMalloc` / VMM / host-location VMM / anon /
registered anon, with `O_DIRECT` and the register-then-fill trap), and the harness lessons in
`MEASUREMENTS.md` §harness.

## Do not do these again

- **Do not port the discrete tier to UMA.**  Arena + copies + eviction is a *discrete* answer to a *discrete*
  problem (VRAM is 13x the link).  On UMA the medium is the same on both sides.
- **Do not size an arena from "free VRAM"** on a shared pool; and do not expect a small arena to exist at all —
  the slab's work/floor/ring requirement is ~9 GiB for a 122B-class model.
- **Do not register a tier before filling it** (70–210 MB/s), and do not force `MOE_HOST_POOL_RO=1` with a live
  arena *and* sink.
- **Do not treat "identical to 10.9 t/s" as "the pool matched the CPU path"** — check `table_pool_locked` /
  the teardown census first; three arms in this campaign were silently streaming.
- **Do not trust the exit census to print.**  `llama-server` under `SIGTERM` printed it in 1 of 6 arms; use
  `llama-cli` for a census, or add a mid-run report.
- Text arms take **no `-lv`**, censuses take `-lv 4`, `-lv 3` is the trap (`AGENTS.md`).
