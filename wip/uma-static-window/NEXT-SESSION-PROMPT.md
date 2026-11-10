# NEXT-SESSION-PROMPT — `wip/uma-static-window`

Hand this verbatim to a cold session.  It assumes no memory of any previous session.

---

You are continuing the `wip/uma-static-window` campaign on this box (`halo` — Ryzen AI MAX+ 395, **gfx1151** UMA APU,
123 GiB, kernel 7.2.13, ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`, 16 cores).

**The campaign is at a DECISION POINT, not a debugging point.**  The zero-copy window is now *correct and free*
(measured, two prompts, byte-identical to the reference), and the arms that were supposed to decide the campaign are
**unreachable by construction** on this box: a window cannot be *partial*, and a window that covers everything never
fits where it would help.  What is open is whether to (a) close the campaign as a negative with its kept results, or
(b) fund the one mechanism that could lift the constraint (a third location in the cold-id space, or unregistered
pageable reads — see "The two forks" below).  Both are the maintainer's call.

## Read first, in this order

1. `~/llama-cpp-rdna-boosts/AGENTS.md` (policy: apply/push/gates/default-on/report-before-commit).
2. `~/llama-cpp-rdna-boosts/wip/uma-static-window/README.md` — the design as amended, and the **P0/P1 verdict block
   (six findings)** plus **P2 verdict finding 7** (the window works; it cannot be partial).
3. `~/llama-cpp-rdna-boosts/wip/uma-static-window/p0/RESULTS.md` — **§L** (the P2 fix, the numbers, the constraint,
   the unregistered-pages probe) is current.  §H is the arithmetic, §K is the superseded "data is wrong" note
   (read it for what the defects were NOT), §G is superseded.
4. `~/llama-cpp-rdna-boosts/archive/work/apu-expert-pool/CONCLUSION.md` — why the arena thesis died (its 122B
   numbers are CORRECTION-flagged as having no arm behind them; do not quote them).

## State

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`apu-expert-pool`** — commit and `git push origin apu-expert-pool` when you change it |
| fork | `~/llama.cpp`, branch **`wip/uma-static-window`** (r243 delivery + the `apu-expert-pool` enablement/fixes + the P0 RO hunk + the P2 window + the P2 layout fix).  Never push upstream. |
| build | `cd ~/llama.cpp && cmake --build build-wip -j 16` (or `BUILD_DIR=build-wip ~/bin/build-llama-rocm`).  **Never** touch `build-rocm`. |
| delivery set | **nothing in `patches/`; `release.json` untouched; no default changed.** Everything is behind opt-in env values (`MOE_HOST_POOL_ALLOC=window`, `MOE_HOST_POOL_WINDOW_VERIFY`, `MOE_HOST_POOL_WINDOW_ORDER`). |

## What is established (measured, with the harness in `p0/`)

**The transport is free.** On this box `hipMalloc` is GTT — the same DRAM as host RAM — and a registered read-only
window reads **231.0 GB/s against hipMalloc's 232.1** at 320 routed rows, **101.9 against 90.3** at the 8-row/1024-thread
shape (`p0/regprobe.cpp`, `p0/band.cpp`).  The band is set by the **launch shape**: 8 rows/launch = 90-105 GB/s,
≥32 rows = 205-234.

**The zero-copy window is CORRECT and FREE** (RESULTS §L.4).  35B `-ncmoe 40`, `-n 96 seed 42 temp 0`, no `-lv`:
**377 chars `sha=ae39ec41bc1e`** — the reference text — heat-ranked, **59.2 t/s**, `fills=0 evictions=0`,
**cold 99.61 %**, 18662 MiB pinned **for ~0 MiB of pinned RAM** (the copy pool spends the 18.6 GiB for 59.8-60.1).
Second prompt `prompts/code-reasoning-mixed.txt`: `sha=ff86b6990ce7` on both.  `p0/arms.sh window-pool 96`.

**A window cannot be PARTIAL** (RESULTS §L.5) — this is the constraint that decides the campaign.  The cold-id space
has exactly two levels (arena slot, or `n_res + cold index` resolved against ONE base and ONE stride), so an op with
some experts in the window and some in the page cache is unrepresentable.  Measured on the 122B at 88 GiB: the window
builds (`206/256` slots/table, 94554 MiB, `fills=0`) and the run decodes at **0.9 t/s** with
`device remap: 0 tables served, 9360 of 9360 offers declined`.  Covering the complement with an *arena* costs 23 GiB
of copy beside the 88 GiB window — over budget before the 8.6 GiB of weights.

**So: 122B decode at 96/88/80 GiB and 122B prefill are NOT reachable on this box**, and neither is the capacity rig's
(its 63 GiB expert set would have to be pinned whole).  Every path is now: fits → a window equals a device placement
(P0: same DRAM, same band) → use `-ncmoe`; does not fit → a window cannot cover its tables.

**Two layout defects were what §K's wrong data was, and neither is visible to a bandwidth probe:**

1. the cold stride is divided by `ggml_type_size` and indexed **in type blocks**, so a pitch of `host_bytes + page`
   truncated (Q4_K: 4124.4 → 4124 blocks = 64 B/slot short, drifting linearly);
2. a pitch of `host_bytes` makes consecutive slots **share a page**, coherent only when `rank[s+1] == rank[s]+1` —
   never for a heat ranking (measured: right to byte 588000 of 589824, wrong for the last 1824, on 241/256 slots).

The pitch is now the smallest multiple of `lcm(page, ggml_type_size(tensor->type))` that holds the expert plus its
file-offset residue (35B 589824 → 626688; 122B 3342336 → 3411968), **registered per slot** (the padding is
`PROT_NONE` and is never pinned), and the window **refuses** when a tensor's stride is not a page multiple or its type
block size is unknown.  `type_blk` is plumbed through `moe_cache_table(..., type_blk)`.

**Baselines on this box** (`-t 12`, `HIP_VISIBLE_DEVICES=0`, no MTP):

| case | number |
|---|---|
| 35B `-ncmoe 0` (all device) | 62.7 t/s · text `ae39ec41bc1e` at `-n 96, seed 42, temp 0` |
| 35B CPU expert path (`-ncmoe 40`, cache off) | 35.0 t/s · same text |
| 35B copy-pool 99.5 % cold | 59.8-60.1 t/s · same text |
| **35B window, full coverage** | **59.2 t/s · same text · `fills=0` · ~0 memory** |
| 122B all-host CPU path | **8.0 t/s** marginal (tokens 64..320), 5.7 (320..600) |
| 122B with 57.4 GiB of experts on the device | 8.5 t/s |
| 122B prefill, 4059 tokens, `-ub 4096`, cache armed | **175.2 t/s** (23.17 s), of which **9.98 s is the fill memcpy** (33.15 GiB at 3772 MiB/s) |

## The two forks (the maintainer decides; do not build either unasked)

**Fork A — a third location in the cold-id space.**  `moe_cache_get_cold` returns one base + one stride; the kernel
computes `cold_base + (id - n_res)*stride`.  A partial window needs `id -> (base, offset)` per slot (or a small range
table: "ids 0..K-1 are window slots, K.. are page-cache experts").  That is shared-kernel machinery (mmvq-moe and the
mmq prefill path), and it must keep the pinned-master and copy-pool paths byte-identical.  It is what makes the
campaign's actual design (partial window + page-cache complement) expressible at all.

**Fork B — find out whether the map+register step is necessary.**  RESULTS §L.6: an **unregistered** `MAP_SHARED`
file mapping is device-readable on this box at the same band as a registered read (4/4 runs, byte-exact vs `pread`),
while fresh **anonymous** pageable memory **faults** every run.  If that holds for the model mapping under reclaim,
then a partial window is moot (the complement is the same pages, read by both), the window costs nothing at all, and
the pinning budget stops being the design's boundary.  **What is not known:** what the GPU does when one of those
pages is **reclaimed mid-kernel**.  A GPU fault is a queue hang, not a retry, so the experiment that matters is a
working set larger than RAM: read a cold 20-40 GiB unregistered file mapping with a sampler on `MemAvailable`, and
force reclaim (`echo 3 > /proc/sys/vm/drop_caches` needs root; `posix_fadvise(DONTNEED)` on the range does not).
A fault there is the answer, and it will hang the queue — run it in its **own process**, never beside a measurement.

## Traps (each cost real time)

1. **`cudaMemcpy` out of a host pointer is a CPU memcpy** under UVA: a "device read-back" that matches proves nothing.
   Read through a **kernel** (`p0/regprobe.cpp`, and `MOE_HOST_POOL_WINDOW_VERIFY=1` in the fork do it properly).
2. **A partial pool cannot be filled and must not be written**: a window slot is mapped `PROT_READ`, so any fill into
   it is a **SIGSEGV in `__memmove_avx512`** (that is how the first 88 GiB arm died).  `pool_fill_empty_locked` /
   `pool_slot_locked` / the prefill span fill now refuse for a window; a miss falls through to the bounce ring.
3. **A slab budget below `used + reserve + 2 chunks` declines and the run silently becomes the CPU path**
   (`STREAMING the experts from the host`).  The t/s is then exactly the CPU path's — check the census, never the rate.
4. **`GGML_CUDA_SLAB_MIN_ARENA_MIB` (default 2048) is a hard floor**; a deliberately small arena needs it lowered.
5. **Raising the pool ceiling pins the box into swap** when the pool is a *copy*: 96 GiB pinned + the mapping it was
   filled from = 22 GiB of swap and 0.8 t/s.  A window avoids that by construction.
6. **The UMA pool ceiling defaults to 10 % of `MemAvailable`** and silently scales a planned 96 GiB pool by 0.122 —
   any "the tier cannot hold N" result must first confirm the clamp did not fire (`MOE_HOST_POOL_RAM_PCT=0` lifts it).
7. **Text arms take no `-lv`; a census is a separate `-lv 4` run** (`-lv 3` is the trap).  `HIP_VISIBLE_DEVICES=0` on
   every arm.  Compare text only within one kernel path.
8. **Never leave a workload unsupervised**: own session (`setsid`), `timeout -k 5`, a memory sampler
   (`p0/window122.sh` does all three), kill the group.
9. `check-gpu-faults.sh` **on every arm** — and note that `p0/regprobe.cpp`'s unregistered arms deliberately fault the
   queue (`read_rows_bw`), so journal faults from a probe run are the probe's, not an arm's.

## Harness (all in `wip/uma-static-window/p0/`, logs in `results/`)

| tool | what |
|---|---|
| `arms.sh <arm> [n]` | 35B arms: `ref-device`, `cpu-path`, `cold100-strict`(+`-wr`), `window5/25/50/75/100`, `cold100`, **`window-pool`** (the P2 window at full coverage — the acceptance arm) |
| `long.sh <arm>` | the same arm at `-n 96` and `-n 512`, with the **marginal** rate |
| `arms122.sh <ncmoe> [n]` | 122B arms with a memory sampler |
| `window122.sh <GiB> [n] [decode\|prefill]` | **122B window arms** (bounded, `setsid`, sampler) — currently reachable only as a swap-containment / decline demo |
| `p1.sh <arm> [n]` | capacity rig arms (ballast in its own session) |
| `cpuwatch.sh <arm>` | CPU % of one core + GPU busy % |
| `band.cpp` | the window band probe (`dev`/`anon-ro`/`anon-rw`/`file-ro`, rows × block size) |
| `regprobe.cpp` | the **content** probe: registered and **unregistered** (file vs anonymous) device reads, plus the band of each |
| `winprobe.cpp` | the zero-copy window builder (VA + `MAP_FIXED` + register) and its band |
| `prior.py` | the achievable hit rate from the `MHT2` sidecar at any resident budget |

Commit and push every delivery-repo change to `origin apu-expert-pool`; report findings to the maintainer before
changing any default, and do not put anything in `patches/` without a validated win on this box.
