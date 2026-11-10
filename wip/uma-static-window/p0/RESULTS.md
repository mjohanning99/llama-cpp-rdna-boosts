# P0 — the three questions, answered by measurement (2026-11-10)

**Verdict, one line each:**

1. **P0.1 — PASS, and better than expected.**  99.5 % of routed expert accesses served **in place** from a
   registered **read-only** pool with a 1-slot/table arena: **59.8 / 60.1 t/s**, against **62.7 / 62.9** for
   every expert on the device and **35.0 / 35.1** for the CPU expert path.  The in-place read is **free**
   (−4.5 % vs device, **+71 % vs the CPU path**) — the machinery the campaign wants already exists and costs
   nothing.
2. **P0.2 — one routed op DOES mix window + in-place experts (free); it CANNOT mix GPU with CPU-computed
   experts, and the alternative granularity is a cliff, not a slope.**  A per-expert window + cold complement
   inside the GPU is free (5 %→75 % window: 53.3/55.6/58.3/59.3 t/s).  But **any table that gets no GPU
   representation falls to the CPU and the whole decode collapses to CPU parity** (81/120 tables GPU-served →
   34.5; 96/120 → 35.5; both = the CPU path's 35.0).  The split is **per table/layer, not per expert**, and its
   cost is not proportional.
3. **P0.3 — PASS.**  A captured decode graph replays with the cache remap in the op stream: `graphs reused = 94`
   for a 96-token decode, `graph reallocs = 4 / 0`, `device remap: 11640 tables served, 0 declined`, no per-token
   host readback, and **byte-identical text** to the all-device arm.  Publishing the map once instead of per op is
   an optimisation, not a requirement: the per-op remap costs ≤1 % (59.8 vs 62.7 with no remap at all).

**The stop rule is not triggered** (P0.1 is 1.7x the CPU path, and P0.2's mixed op is at device parity), **but the
campaign's arithmetic has to be rewritten** — see §5.  The risk is no longer "can the GPU read where the bytes
are"; it is "how much of the set can be *registered* on this box, and what does the CPU complement cost".

---

## 1. The rig and the arms

`halo` (Ryzen AI MAX+ 395, gfx1151 UMA, 123 GiB, ROCm 10.1.0-gfx1151), `build-wip` built from the r243 delivery
tree + the five `wip/apu-expert-pool` enablement/fix commits (`70e0eda4a`; the retired experiment `9b717f29a` is
**not** in it), plus **one P0-scoped hunk** (see §6).

Model: `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf` — 21.1 GiB, 40 layers, 120 host-expert
tables, **18 662 MiB of expert bytes** (0.59–0.72 MiB per expert, 256 experts/table).  A `.moe-heatmap` prior is
present from the previous campaign.

Every arm: `HIP_VISIBLE_DEVICES=0`, `-ngl 99 -c 4096 -t 12`, prompt `The capital of France is`, `-n 96`,
`--seed 42 --temp 0 --single-turn --no-display-prompt`.  Text arm **no `-lv`**, census **a separate `-lv 4` run**
(`p0/arms.sh`).  No two arms ever ran concurrently.

Harness: [`arms.sh`](arms.sh) · logs: [`results/`](results/) · `scripts/check-gpu-faults.sh`: clean (kernel
journal + all three sampled logs).

## 2. The numbers

| arm | what | decode t/s | tables GPU / CPU | residency / cold | text |
|---|---|---|---|---|---|
| `ref-device` | `-ncmoe 0`, everything on the device | **62.7 / 62.9** | — | — | `ae39ec41bc1e` |
| `cpu-path` | `-ncmoe 40 --host-experts pool -lm mmap`, cache off | **35.0 / 35.1** | 0 / 120 | CPU in place from the page cache | `ae39ec41bc1e` |
| `cold100-strict` | all tables registered, **1 slot/table**, admission never admits, **RO** pool | **59.8 / 60.1** | 120 / 0 | **0.4 % resident, 99.48 % cold** | `ae39ec41bc1e` |
| `cold100-strict-wr` | the same, pool registered **writable** | 59.9 / 59.9 | 120 / 0 | 99.49 % cold | `ae39ec41bc1e` |
| `window5` | 5 % of each table resident, rest cold | 53.3 / 53.4 | 120 / 0 | 31.8 % hit, 54.6 % cold, 13.6 % fill | `ae39ec41bc1e` |
| `window25` | 25 % | 55.6 / 55.7 | 120 / 0 | 70.7 / 20.1 / 9.1 | `ae39ec41bc1e` |
| `window50` | 50 % | 58.3 / 58.1 | 120 / 0 | 94.6 / 2.1 / 3.3 | `ae39ec41bc1e` |
| `window75` | 75 % | 59.3 / 59.3 | 120 / 0 | 98.9 / 0.0 / 1.1 | `ae39ec41bc1e` |
| `cold100` | arena 528 MiB (slab cap 6 GiB): 39 tables got no arena | 34.5 / 34.5 | **81 / 39** | 3.7 / 94.9 | `ae39ec41bc1e` |
| `window100` | `SLOTS=256`: the arena ran 0.4 GiB short, 24 tables got no arena | 35.5 / 34.6 | **96 / 24** | identity fast path on 96 tables | `ae39ec41bc1e` |
| `cold100` (cap 24 GiB, budget-derived slots) | full mirror residency | 58.5 / 58.7 | 120 / 0 | h=0.71, 12.8 % cold | `ae39ec41bc1e` |

Text is **identical in every arm** (377 chars, `sha=ae39ec41bc1e`), including the arms whose text changed with
`-n`, which is the point: the in-place read, the mixed op and the CPU split are all bit-identical, as expected
(they move no arithmetic).

## 3. P0.1 — 100 % in-place pool service

`cold100-strict` is the arm the question asks for: an explicit slab budget big enough that **all 120 tables
register** (24 GiB cap → 20.67 GiB of slab), `MOE_EXPERT_CACHE_SLOTS=1` (one arena slot per table = 72.9 MiB,
0.4 % of the set), `MOE_EXPERT_CACHE_TOUCH=1000000` (the admission never admits), a **covering mirror pool**
(`MOE_HOST_POOL_FRAC=100`, 18 662 MiB pinned) registered **read-only**.

Census: `expert access: hits=373 fills=110 colds=93061 (hit=0.40% fill=0.12% cold=99.48%)`,
`cold admission=touch, admission-rejected (served cold)=88856`, `device remap: 11640 tables served, 0 declined`,
`takeover=11760 decline_all=0 get_ok=600`.

* **59.8 / 60.1 t/s against the CPU path's 35.0 / 35.1 → +71 %.**  This is the campaign's core premise, measured
  for the first time: **the GPU reading host-resident experts where they are, with no copy and no eviction, is
  not a compromise.**
* Against the all-device arm (62.7) the cold path costs **−4.5 %** — i.e. 95.5 % of having the bytes in device
  memory, which on a UMA APU is the *same DRAM*.  There is nothing left for a copy tier to win on this box.
* **Read-only vs writable registration is a tie here** (60.1 vs 59.9).  That is a *methodology* result, not a
  refutation of the 233-vs-60 GB/s probe: at batch 1 a 35B-A3B decode moves 777 MiB/token in 16.6 ms ≈
  **46 GB/s of effective expert bandwidth**, far below either registration's ceiling, so the 35B **cannot**
  resolve a read-band question.  It can only prove the path is cheap — which it did.

## 4. P0.2 — the split, and its granularity

**Within the GPU the mix is free.**  `window5..75` are one routed op per layer carrying both encodings
(`slot` for a resident expert, `n_res + pool slot` for the rest) — the id encoding the design assumed.  The
curve is smooth and monotone (53.3 → 59.3) and every arm is above the CPU path by 1.5x.  So *"can one routed op
mix window-resident experts with ones served from the pool in place"* is **yes, at no measurable cost**.

**Crossing to the CPU is where it breaks.**  An op runs on one backend, so "some experts on the GPU, the rest
computed on the CPU" cannot be one ggml op.  The delivery's expression is per **table** (a table with no arena is
simply not taken over, and its layer's MoE stays on the CPU).  Measured:

| arm | tables on the GPU | decode |
|---|---|---|
| all tables GPU-served (any window fraction) | 120 / 120 | 53.3 – 60.1 |
| `window100` (arena 0.4 GiB short) | 96 / 120 | **35.5** |
| `cold100` (arena 512 MiB) | 81 / 120 | **34.5** |
| CPU expert path | 0 / 120 | **35.0** |

Two conclusions, both load-bearing:

* **The split is a cliff.**  Losing 20 % of the tables costs *the entire* GPU advantage; losing 32 % costs no
  more.  It is not proportional to the CPU fraction, so it is not "the CPU part is slow" — it is a fixed
  per-token cost of having a CPU-side MoE chain at all (scheduler split boundaries; the decode graph's `splits =
  42` was identical in every arm, so the split structure, not the op count, is what changes).
* **"The split is per expert, which is what the heat prior ranks" (README §2) is not what the machinery does.**
  Per-expert works *inside* the GPU (window vs pool-cold).  Per-expert **GPU vs CPU** does not exist anywhere in
  the delivered path; the only GPU/CPU granularity is the table (or, in the upstream `--moe-cache-mib` cache,
  `cpu_share` — a different cache, dormant by default, and it pays the same two-chain cost).

**What this does to the arithmetic.**  The design's 2–3x row assumes the window read and the CPU read *overlap*
(`194f + 41(1−f)`).  Nothing measured here overlaps: the CPU chain adds its time.  So the honest model for a
>RAM model on this box is additive, and it is bounded by how much can be **registered** — because "registered"
(pinned) and "page cache" come out of the same 123 GiB.

Worked on the 122B (114.8 GiB of expert bytes, 3.8 GiB routed per token, CPU path 10.9 t/s = 41 GB/s):

| window (pinned) | additive time/token (GPU 194 GB/s + CPU 41 GB/s) | predicted |
|---|---|---|
| 20 % (23 GiB) | 16.5 + 75.9 = 92 ms | ~10.9 t/s (**no gain**) |
| 40 % (46 GiB) | 8.3 + 55.6 = 64 ms | ~15.6 t/s (+43 %) |
| 60 % (69 GiB) | 5.5 + 37.1 = 43 ms | ~23 t/s (+111 %) |

and every row above assumes the pinned window does **not** evict the page cache the CPU complement reads from —
at 46 GiB of pins against a 123.4 GiB model on a 123 GiB box, that assumption is already gone.  **So the campaign's
2–3x is not reachable on this box; a realistic target is +10–40 % at 20–40 % window, and the design must not
assume overlap.**

## 5. What this changes about the campaign

1. **The static window is a *sizing* feature, not a *transport* feature.**  The transport (in-place RO read) is
   already here and already free.  What the window buys is predictability: a heat-ranked resident set chosen once
   instead of an LRU that fills, evicts and copies during the run (the window arms did 1 000–17 000 fills per
   96 tokens — that is the delivered tier's remaining cost, and it is exactly what a static window removes).
2. **All-or-nothing per table is a hard requirement.**  A window that leaves *one* table without a GPU
   representation loses the whole win.  So window sizing must never overshoot the arena, and the campaign needs a
   census assertion for it (`tables == n_host_tables`), not just a "did the slab arm" check.
3. **The complement must be GPU-served or the win is gone.**  On a box where the model is bigger than RAM that is
   impossible for the whole set; the design should be aimed at the *fraction that can be pinned*, with the rest
   accepted at CPU parity, and the target restated as "the pinned fraction runs at device speed".
4. **The read-band question is open and the 35B cannot answer it.**  It needs either the 122B arm (long) or a
   ~80-line HIP probe of the real access pattern (8 × 0.6 MiB random reads from an RO-registered 18 GiB buffer),
   because the campaign's per-token arithmetic depends on whether that number is 46, 60, 194 or 230 GB/s.
5. **`--lme N` as GiB of *window*** fits the measurements: it must be spent on **whole tables** (the granularity
   the split actually has), ranked by the heat prior's per-table concentration, not per expert.

## 6. The one scaffolding change this needed

`moe-expert-cache.cu` (`pool_mmap_thp_pin`): at `70e0eda4a` a read-only pool registration is refused whenever
the arena is *enabled*, on the grounds that the arena→pool demote may write the backing from the device.  The
demote is the only such writer, so the guard is tightened to `g_enabled && g_pool_sink` (taken from the retired
branch's own refinement), which makes `MOE_HOST_POOL_SINK=0 MOE_HOST_POOL_RO=1` the isolation arm the P0 read
needs.  Campaign-branch only; **not** folded into `patches/`, and it changes no default.

## 7. Reproduce

```bash
cd ~/llama.cpp && git checkout wip/uma-static-window          # r243 tree + apu-expert-pool enablement/fixes
BUILD_DIR=build-wip ~/bin/build-llama-rocm                    # or: cmake --build build-wip -j 16
cd ~/llama-cpp-rdna-boosts/wip/uma-static-window/p0
./arms.sh ref-device 96 ; ./arms.sh cpu-path 96 ; ./arms.sh cold100-strict 96
./arms.sh window5 96 && ./arms.sh window25 96 && ./arms.sh window50 96 && ./arms.sh window75 96
```

---

# Addendum (same day): the band probe, long generations, and the 122B

Three follow-ups the maintainer asked for.  Two of them change the campaign's *target case*.

## A. The window's effective bandwidth — `band.cpp`

The question (§9.4 above): what does a GPU pull from a registered window when the reads look like a routed
MoE decode — a handful of random per-expert blocks per op — rather than a 1 GiB sequential sum?

`hipcc -O2 --offload-arch=gfx1151 -o band band.cpp`, 8 GiB footprint, random expert per row, one kernel
launch per "op" (one block per routed row, `__ldg` loads, median of 40 after a warm-up).  `dev` = `hipMalloc`
(the GTT buffer the arena is carved from); `anon-ro` = anonymous mmap filled **then** registered
`Mapped|ReadOnly` (the design's window).

**Expert 610 352 B (the 35B's Q4_K_M), GB/s:**

| reads per op | one launch, `dev` | one launch, `anon-ro` | 8-block launches, `dev` | 8-block launches, `anon-ro` |
|---|---|---|---|---|
| 8 | 93.8 | **93.0** | 95.4 | 93.0 |
| 32 | 204.8 | **204.5** | 100.2 | 98.2 |
| 128 | 228.0 | **227.4** | 99.9 | 99.5 |
| 320 | 232.1 | **231.0** | 99.5 | 100.0 |

**Expert 3 300 000 B (the 122B's Q8_0), GB/s:** 8 rows 103.4 / **101.4**; 32 rows 223.9 / **223.4**;
128 rows 235.3 / **234.8** (one launch; 8-block launches stay at 101–103).

**Block size is the per-launch lever** (anon-ro, 610 352 B): 8 rows × 256 thr = 95.5, × 512 = 142.6,
× 1024 = **176.7**; 16 rows × 1024 = 206.9; 32 rows × 256 = 204.4.

Three facts, and they are what the per-token arithmetic needed:

1. **A registered read-only window reads at device-memory speed.**  231.0 GB/s against `hipMalloc`'s 232.1 at
   320 rows, 227.4 vs 228.0 at 128 — **identical at every shape measured**.  On a UMA APU a window is not a
   slower tier than the device; it is the same DRAM with an extra copy in front of it.
2. **The band is set by the launch shape, not the backing.**  Eight routed rows per launch top out at
   **~95-103 GB/s**; ≥32 rows reach 205–232.  A batch-1 decode issues one op per layer with 8 rows, so **the
   per-layer cold read runs at ~100 GB/s, not 232** — and a 1024-thread block lifts even that to 177.
3. **The read-only registration flag is irrelevant to the read** (probe `03`'s 60 GB/s writable arm did not
   reproduce: 230.3 GB/s here).  The old 60-vs-233 GB/s split is a property of that probe's load pattern, not
   of the registration — which is exactly why `cold100-strict` and `cold100-strict-wr` tie (60.1 vs 59.9).

## B. Long generations — does the hot set establishing itself show up?

Same arm at `-n 96` and `-n 512`; the **marginal** rate over tokens (96, 512] is the steady state.

| arm | n=96 | n=512 | marginal | gain |
|---|---|---|---|---|
| `ref-device` | 62.9 | 63.1 | 63.2 | +0.5 % |
| `cpu-path` | 35.0 | 35.2 | 35.2 | +0.6 % |
| `cold100-strict` | 59.9 | 60.0 | 60.0 | +0.2 % |
| `window25` | 55.5 | 56.2 | **56.4** | **+1.6 %** |
| `window50` | 58.0 | 58.4 | **58.5** | **+0.9 %** |
| `window75` | 59.5 | 59.5 | 59.5 | 0 % |
| `window25`, **no prior** (`MOE_EXPERT_HEATMAP=0`) | 54.9 | 55.8 | **56.0** | **+2.0 %** |
| `window50`, **no prior** | 52.8 | 54.6 | **55.0** | **+4.2 %** |

**The hypothesis is right in direction and small in size.**  A cold start (no persisted prior) gains +2.0 % /
+4.2 % of decode between the first 96 and the next 416 tokens — real, monotone and larger at the smaller
window.  But it never catches the **seeded** arm: `window50` cold starts at 52.8 and converges to 55.0, while
the persisted prior puts the same arm at **58.0** from token 1.  That is the campaign's static-window premise
measured directly: **the persisted heat prior is worth ~+6 % at 96 tokens, more than the whole in-run
convergence**, and it is already implemented (`apply_heat_seed_locked` seeds the arena from the prior at
sizing).  The long-run effect does not change the ranking or close any gap.

## C. The 122B — and why this box is the wrong target case

On this box `hipMalloc` is GTT — **the same DRAM as host RAM at the same speed** (§A.1).  So "pin a window of
the hottest experts" and "put those experts on the device" are the *same memory at the same speed*, and the
second needs no copy.  The only thing that varies is **capacity**, and Qwen3.5-122B-A10B Q8_0 is 123.4 GiB
against ~116 GiB of usable memory here: the model is bigger than the box.

Measured (`-c 8192`, cache off, `--host-experts pool -lm mmap`, marginal over tokens 64..320 so the page
cache state is amortised):

| arm | device model | host model | steady-state decode |
|---|---|---|---|
| `-ncmoe 48` (all experts host) | 5.5 GiB | 118.3 GiB | **8.0 t/s** |
| `-ncmoe 24` (57.4 GiB of experts **on the device**) | 63.0 GiB | 59.5 GiB | **8.5 t/s** |

Moving 57 GiB of expert bytes onto the GPU is worth **+7 %, i.e. nothing**, because every non-reclaimable
device byte displaces exactly one page-cache byte of the part that stays host-resident — and then the CPU
part streams from NVMe.  (The whole-run numbers swing 4.2–9.6 t/s *between* arms purely on page-cache state,
which is why only the marginal rate is quoted.  Neither configuration reaches the recorded 10.58–10.90, which
was taken on a fully warm page cache.)

**The 122B's 10.9 t/s is a capacity wall, not a reader-band wall.**  3.8 GiB/token at the CPU's 41 GB/s is
93 ms = 10.8 t/s — the CPU is already at the only rate the bytes' *residency* allows, and the bytes cannot be
moved closer to the GPU because there is nowhere to move them: on UMA the device pool and the page cache are
one resource.

## D. Where that leaves the campaign

The design is not refuted — the transport is free (§ above) and the static prior beats in-run convergence
(§B) — but its **home is the capacity-constrained case, not the >RAM one**:

* a model that **fits in RAM but not in the device share** (or a box where the two are separate), where
  pinning a window takes bytes *out of the page cache* only in the sense of making them non-reclaimable, and
  the win is the reader (231 GB/s against the CPU's 41);
* which on this box is exactly the **56 GiB ballast rig** (`wip/bench-56gb/`), where the copy tier already
  measures IQ4_NL at **44.0 t/s against the CPU path's 31.0 (+42 %)** — a configuration the campaign can
  reproduce and, if the static window is worth anything, beat (no per-miss copy, no eviction, no per-op
  remap), while the >RAM 122B has no slack to give.

**Next (P1):** the ballast rig, IQ4_NL, static-window arm vs the copy tier vs the CPU path — plus the
launch-shape lever from §A.2 (a 1024-thread routed block is worth 177 vs 95 GB/s at 8 rows, which is free
bandwidth for whoever owns that kernel).

## E. "Is the CPU still doing the expert work?" — checked three ways (maintainer's question)

The `cold100-strict` census reports 99.48 % cold, but the log carries
`ggml_backend_cpu_graph_n_threads: offloaded MoE (host-resident expert weights): using 8 of 12 threads` in **every**
host-expert arm, and the graph still has 120 scheduler splits — so "is the CPU computing MoE here?" had to be
answered directly rather than assumed.  `ggml_backend_cpu_graph_n_threads` scans the **CPU split** for a
`MUL_MAT_ID` whose weight is a *host* buffer owned by another backend (`ggml-cpu.cpp:207-218`), i.e. exactly a
host-expert tensor, so the line is real evidence that such nodes exist somewhere in the run — but it is logged
once per process, so it cannot say *how many*, and the prompt-eval/load graphs alone can trigger it.

Three measurements (`p0/cpuwatch.sh` samples `/proc/<pid>/stat` and `rocm-smi`; the `-t` arms re-run the same
arm):

| arm | mean CPU (of one core) | mean GPU busy | decode | `-t 4` | `-t 1` |
|---|---|---|---|---|---|
| `ref-device` | 65.9 % | 21.4 % | 62.6 | — | **62.8** |
| `cpu-path` | **759.3 %** | 42.3 % | 35.9 | 33.8 | **26.7** |
| `cold100-strict` | **142.9 %** | 21.9 % | 60.0 | 60.0 | **59.8** |

**The cold arm is not on the CPU expert path.**  It is completely **thread-insensitive** (60.0 at `-t 12`, 60.0 at
`-t 4`, 59.8 at `-t 1`) against the CPU path's 35.9 → 26.7, and its CPU is 143 % against the CPU path's 759 %.  The
GPU does the expert math; `takeover = 11760` (all 120 tables × 96 tokens) matches the op count, and the decode rate
equals the all-device arm's.

**But the observation was not empty:** the cold arm spends **~77 % of one core more than `ref-device`** (142.9 vs
65.9) on the same graph — the per-op cache bookkeeping (`access_locked` per expert, the cold-id encoding, the remap
publish).  That is the "per-op host round-trip" the design exists to remove, it is measurable, and it is the one
place where the CPU is still in the loop.  It does not bind on a 16-core box; it would on a smaller one.

**A second, unexpected result in the same table:** the CPU path itself is only **1.35x** slower at `-t 1` than at
`-t 12` (26.7 vs 35.9).  So the 35B's CPU expert path is **latency-bound, not thread-bound** — which is why the GPU
wins only 1.7x there and why this model cannot be used to size a bandwidth argument at all (§A, §C).

---

# F. P1 -- the capacity-constrained rig (IQ4_NL, 48 layers, 512 experts, 10 used, 63 GiB of experts)

The rig: `wip/bench-56gb/ballast 31` (mlock'd, own session, MemAvailable 120 -> 88 GiB), IQ4_NL Flash-Next,
`-ncmoe 24` (host expert set **31.6 GiB**, device model 34.4 GiB), `-c 8192`, `-n 384`, no MTP drafter, `-t 12`.

**The recorded references do not survive measurement.**  The closed campaign's "44.0 t/s tier / 31.0 t/s CPU path"
figure exists in exactly one place -- the same `CONCLUSION.md` that carries the unverified 122B numbers -- and
`wip/bench-56gb/RESULTS.md` has no IQ4_NL arm (that rig ran GSQ-RCO-IQ3_S).  Measured here, same rig, same protocol:

| arm | what | decode (n=384) |
|---|---|---|
| `cpu-path` | `-ncmoe 24`, host experts in the page cache, cache **off** | **14.8 t/s** |
| `tier21` | `-lme 21`, the **shipped tier** (arena 24.1 GiB + pool 21.5 GiB, h = 0.925) | **26.3 t/s** |

So the *qualitative* claim in the closed record holds — a tier is worth **+78 %** in the capacity-constrained case —
while the absolute numbers are 1.7–2.0x below the unverified ones.  (The two arms' greedy text differs, `0584eff2` vs
`83d058e8`: expected and not a defect — the CPU path computes those experts with the CPU kernels and the tier with
the GPU's, and text identity is a per-kernel-path property, not a cross-backend one.)

**The static window cannot be measured in this configuration, and that is the finding.**  The tier's win above is
**capacity**, not transport: 24.1 GiB of arena **plus** 21.5 GiB of pool covers 45.6 GiB of a 63 GiB expert set
(h = 0.925).  A static window has to *cover every table it serves* (`pool_covers_table`: `slots >= n_experts`) or its
cold ids are not valid — so a covering window on this rig means pinning the whole 31.6 GiB host set, and
31 (ballast) + 34.4 (device) + 31.6 (pinned) = 97 GiB against 88 GiB available.  It does not fit.  At a budget that
*does* fit (21 GiB), the delivered pool spreads ~340 of 512 slots per table, i.e. **no table is covered** and the
window shape degenerates to the shipped one.

**And a partial window (some tables GPU, the rest CPU) is predicted to lose**, from P0's two results: the split is
per table, and its cost is a cliff.  On this model a 21 GiB whole-table window covers ~11 % of the layers, and P0's
`window100`/`cold100` arms show that any CPU-side MoE chain drags the decode to the CPU path's rate.  That is a
prediction from P0, **not** a measurement — making it a measurement needs the per-table placement policy that does
not exist yet (a P2-sized change).

**So, after P0 + P1, there is no configuration on this box where the static window beats what already ships:**

| regime | measured | the window's place |
|---|---|---|
| fits (35B, 21 GiB of 116) | window = **device parity** (59.8–60.1 vs 62.7), +71 % over the CPU path | works, but on UMA it *is* a device placement plus a copy — `band.cpp` shows 231.0 vs 232.1 GB/s |
| does not fit (122B, 123.4 GiB) | 8.5 vs 8.0 t/s with 57 GiB moved to the device | nothing: no slack, the box is the constraint |
| capacity-constrained (IQ4_NL + 31 GiB ballast) | shipped tier **26.3** vs CPU path **14.8** | cannot be built at a budget that fits (covering window = whole host set pinned) |

The two results worth keeping regardless of the campaign's fate: **the window read is device-speed and its band is
set by launch shape** (§A), and **the 122B/44.0 reference numbers had no arm behind them** (§C, §F).

---

# G. The 122B question, answered as a bound rather than a hope

**Question (maintainer, 2026-11-10): is there *any* path on which the campaign runs Qwen3.5-122B-A10B Q8_0 faster
than stock llama.cpp's default on this box?**

**Answer: no — and the bound is not the campaign's own measurement, it is stricter than that.**  The decisive arm
already exists: `-ncmoe 24` puts **57.4 GiB of the 114.8 GiB expert set on the device** (f = 0.5), where the GPU
reads it at 232 GB/s **with no copy, no registration, no fill, no eviction and no per-op remap**.  That is a
*strict superset* of what the campaign's window can offer — every benefit the window claims, none of its costs — and
it measures **8.5 t/s against the all-host arm's 8.0 (+6 %)**.

So **+6 % is an upper bound for the campaign on this model**, and a pinned window can only be equal or worse (it
adds a copy and a registration to the same bytes, and `band.cpp` shows the registered window reads at exactly
hipMalloc speed).

**Why the bound is so low — the shortfall is invariant to placement.**  The 122B is 123.4 GiB (114.8 experts + 8.6
non-expert) against ~116 GiB usable here: **7.4 GiB short**.  Pinning `f` of the experts takes `f` out of the page
cache at the same time, so the *streaming share of the complement rises*, it does not fall:

| f (pinned) | complement | page-cache room | share streaming from NVMe |
|---|---|---|---|
| 0.0 | 114.8 GiB | 107.4 GiB | 6.4 % |
| 0.2 | 91.8 GiB | 84.4 GiB | 8.1 % |
| 0.4 | 68.9 GiB | 61.5 GiB | 10.7 % |

An additive model that *assumed* the CPU side kept its 30 GB/s band would predict +37 % at f = 0.4 and +51 % at
f = 0.4 with a 231 GB/s window — the table above is why that prediction is unavailable: the CPU side's band comes
from its cache hit rate, and the window lowers it.  The measured +6 % at f = 0.5 is what remains.

**The general rule this produces:** on a box where device memory and host RAM are **one pool**, a window changes
*who reads* the bytes and never *how many bytes need reading*.  It therefore only pays where the reads — not the
capacity — are the limit.  For the 122B on this box the capacity is the limit, so nothing placement-shaped can help.
The levers that would actually move it are a smaller quantisation (fit under ~107 GiB of experts), more RAM, or
faster storage for the 6-11 % that streams — not a window.

**Where a yes would exist:** a box where device memory is a *separate* pool, so a window adds capacity on top of
RAM and reads it ~13x faster than the CPU (the shipped tier's whole reason for existing on discrete cards), or a
smaller model where the expert set fits RAM but not the device share (`wip/moe-static-hot-tier`'s territory, and
the reason the campaign's home moved to the capacity rig in §F).

---

# H. The 96 GiB question — the arithmetic is RIGHT, and it amends the design

**Maintainer's argument:** with the hottest ~96 GiB heat-managed, the hit rate is ~98 %, so only ~1 in 50
experts comes from SSD at ~6 GB/s — so decode should run at near-full speed on the hits and slow down only on
the misses.

**Measured from the persisted prior itself** (`p0/prior.py`, which parses the `MHT2` sidecar — no model load,
no arena, no slab): Qwen3.5-122B-A10B Q8_0, 144 tables × 256 experts, 114.8 GiB of expert bytes, 3.19 MiB per
expert, total routed mass 5 029 992 decode-band accesses:

| resident | % of bytes | **hit rate** | miss traffic / token |
|---|---|---|---|
| 8 GiB | 7.0 % | 39.99 % | 2.28 GiB |
| 24 GiB | 20.9 % | 64.86 % | 1.34 GiB |
| 48 GiB | 41.8 % | 85.89 % | 0.54 GiB |
| 64 GiB | 55.8 % | 93.99 % | 0.23 GiB |
| 80 GiB | 69.7 % | 98.42 % | 60 MiB |
| **96 GiB** | **83.7 %** | **99.94 %** | **2 MiB** |
| 104 GiB | 90.6 % | 100.00 % | 0 |

So at 96 GiB the miss traffic is **2 MiB/token** — the SSD term is *nil*, exactly as argued.  And the prior also
shows why `-ncmoe` cannot express it: **the mass per table is uniform to 1 %** (min 34 672, max 34 936), so a
*layer* placement covers its byte fraction of the mass and nothing more.  That is why `-ncmoe 24` (50 % of the
bytes) measured +6 % rather than +50 %: it was a heat-blind placement, not an upper bound for a heat-ranked one.
**§G's "upper bound" was therefore wrong** and is superseded by this section.

**Predicted decode with a 96 GiB heat-ranked window:** 0.9994 × 3.8 GiB/token read at the measured window band —
**99 GB/s at the per-layer launch shape (8 rows/op) → ~26 t/s, 177 GB/s with a 1024-thread block → ~47 t/s, 221 GB/s
for ≥32 rows → ~55 t/s** — against the all-host CPU path's 8.0 t/s measured here.  **3-6x stock**, which is the
campaign's original claim and is now supported by data rather than assumed.

## The two things that broke, and the amendment

**1. The shipped pool cannot deliver it, because the pool is a COPY.**  Run: `-lme 96 --host-experts pool -lm mmap`
with the RAM ceiling lifted (`MOE_HOST_POOL_RAM_PCT=0`; the branch's UMA default is **10 % of MemAvailable =
11.97 GiB**, which silently scaled a planned 96 GiB pool by 0.122 → 26 slots/table → h = 3.8 %, 82 170 evictions,
3.2 t/s — worth knowing on its own).  With the ceiling lifted the pool really did pin **98 226 MiB**, and the box
**swapped 22 GiB** (MemAvailable 3.7 GiB) and decoded at **0.8 t/s**: 96 GiB pinned is 96 GiB *on top of* the
mapping it was filled from, and 96 + 8.6 (weights) + 4.75 (slab) + the complement's pages does not fit a 116 GiB
budget.  Its own hit rate was also only h = 0.84 — the delivered pool spreads ~214/256 slots uniformly per table,
not the heat-shaped allocation `MOE_HOST_POOL_SIZING=1` fits.

**2. So the window must not be a copy — it must be the model's own pages, registered read-only.**  `p0/band.cpp`'s
`file-ro` arm (the one never run until now): **the real GGUF mmap, 8 GiB registered `ReadOnly`, read at 221.0 GB/s
(32 rows/op) and 99.2 GB/s (8 rows/op) — identical to `hipMalloc`.**  Zero copy, so it costs **no extra memory**:
96 GiB of pinned file pages + 8.6 GiB of device weights ≈ 105 GiB of 116, and the complement (18.8 GiB, 0.06 % of
accesses) stays pageable and is read on demand.

**Amendment to the design (README §2, step 1).**  It currently says "their bytes are copied **once** … into an
anonymous, 4 KiB-aligned buffer".  That copy is what makes the configuration 96 GiB *more* than the box has.  The
form that satisfies the maintainer's arithmetic is: **register the hot experts' pages of the model mapping
`ReadOnly`, in place — fill-then-register becomes register-what-is-already-resident**, which the probe's
`file-ro` arm now measures at device speed.  Cost: registration pins (probe §9, ~8.7 s per 46 GiB), the ranges are
per expert (3.19 MiB each, ~30 800 of them at 96 GiB) because a table's hottest experts are scattered through an
expert-major tensor, and a *pageable* fallback remains for the complement.

**What is now genuinely unmeasured and worth building (P2):** per-expert RO registration of the mapping + the
existing heat prior + the existing per-expert cold-read path, then the 122B arm at 96/88/80 GiB.  Predicted 26-55
t/s against stock's 8.0.  The launch-shape lever (§A.2) is worth +80 % of band on its own at 8 rows.
