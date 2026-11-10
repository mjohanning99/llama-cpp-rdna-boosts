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
