# wip/uma-static-window — a heat-ranked, read-only **static window** for host-resident MoE experts on a UMA APU

**Status: OPEN — P0 DONE (2026-11-10), decision below; still nothing in `patches/`.**  This campaign exists
because the previous one ([`archive/work/apu-expert-pool/`](../../archive/work/apu-expert-pool/CONCLUSION.md))
disproved its own thesis.  Read the CONCLUSION first; this file assumes it.

> ### P0 verdict (the full record, arms and censuses: [`p0/RESULTS.md`](p0/RESULTS.md))
>
> **1. The transport is free (PASS).**  99.48 % of routed expert accesses served **in place** from a
> registered **read-only** pool, one arena slot per table: **59.8 / 60.1 t/s** against **62.7 / 62.9**
> with every expert on the device and **35.0 / 35.1** for the CPU expert path — **+71 % over the CPU path,
> −4.5 % against device memory.**  Nothing here is a compromise, and the machinery already exists.
>
> **2. One routed op mixes window + in-place experts for free, but it cannot mix GPU with CPU-computed
> experts — and the delivered split granularity is the TABLE, not the expert.**  A per-expert window inside
> the GPU is smooth (5→75 % window: 53.3/55.6/58.3/59.3 t/s).  But a table that gets no GPU representation
> falls to the CPU and the **whole** decode drops to CPU parity (96/120 tables served on the GPU → 35.5;
> 81/120 → 34.5; CPU path → 35.0).  The split is a **cliff**: it is a fixed per-token cost of having a
> CPU-side MoE chain at all, not a proportional one.  §2's "the split is per expert" is what the *heat prior*
> ranks, not what the graph can express.
>
> **3. The remap survives a captured decode graph (PASS).**  `graphs reused = 94` for a 96-token decode,
> `graph reallocs = 4 / 0`, `device remap: 11640 tables served, 0 declined`, no per-token host readback, text
> byte-identical to the all-device arm.  Publishing the map once is an **optimisation, not a requirement**:
> the per-op remap costs ≤ 1 %.
>
> **4. Addendum (same day, `p0/RESULTS.md` §A-C): the window and the device are the same memory at the same
> speed.**  `p0/band.cpp` measures a registered read-only window at **231.0 GB/s against `hipMalloc`'s
> 232.1** at 320 routed rows, 227.4 vs 228.0 at 128 — identical at every shape — and shows the band is set
> by the **launch shape**, not the backing (8 routed rows/launch = **95-103 GB/s**, >=32 rows = 205-232; a
> 1024-thread block lifts 8 rows to 177).  The read-only flag itself is inert to the read (that is why
> `cold100-strict` and its writable twin tie).  On this box a window is therefore *device memory plus a
> copy*, and the only thing that varies is capacity — which the 122B does not have: `-ncmoe 24` (57.4 GiB
> of experts on the device) measures **8.5 t/s steady state against `-ncmoe 48`'s 8.0**, i.e. nothing, and
> its recorded 10.9 is a **capacity** wall (3.8 GiB/token at the CPU's 41 GB/s).  Long generations confirm
> the static premise but small: a cold start gains **+2.0/+4.2 %** between the first 96 and the next 416
> tokens, while the **persisted prior is worth ~+6 % at token 1** (`window50` 58.0 seeded vs 52.8 cold).
> **So the campaign's home is the capacity-constrained case — the 56 GiB ballast rig, where the copy tier
> already measures 44.0 vs the CPU path's 31.0 (+42 %) — not a >RAM model on a 123 GiB box, which has no
> slack to give.**  P1 target: beat the copy tier there with the static window.
>
> **5. P1 (capacity-constrained rig, `p0/RESULTS.md` §F): the window has no configuration left on this box.**
> IQ4_NL under a 31 GiB ballast, `-ncmoe 24` (31.6 GiB host set), `-n 384`: the **shipped tier is 26.3 t/s**
> against the **CPU path's 14.8** -- its win is **capacity** (24.1 GiB arena + 21.5 GiB pool = 45.6 GiB of a
> 63 GiB set, h = 0.925), not transport.  A static window must *cover every table it serves*, which here means
> pinning the whole host set: 31 (ballast) + 34.4 (device) + 31.6 (pinned) = 97 GiB against 88 available, so it
> cannot be built at a budget that fits.  A partial window is predicted to lose (P0: per-table split, cliff).
> **The recorded "44.0 / 31.0 t/s" references for this model have the same provenance problem as the 122B's
> (CONCLUSION-only, no arm, and `wip/bench-56gb/RESULTS.md` never ran IQ4_NL).**  Net: fits -> the window equals
> a device placement; >RAM -> no slack; capacity-constrained -> the window cannot cover its tables.  Recommend
> closing the campaign as a **negative with two kept results** (the launch-shape band, §A; and the reference
> correction, §C/§F) -- the decision is the maintainer's, and nothing here is in `patches/`.
>
> **6. The 122B question, answered as a bound (`p0/RESULTS.md` §G).**  *Is there any path on which the campaign
> beats stock for Qwen3.5-122B-A10B Q8_0 here?*  **No, and the bound is stricter than the campaign's own arms:**
> `-ncmoe 24` already puts **57.4 GiB of the 114.8 GiB expert set on the device** (f = 0.5) where the GPU reads it at
> 232 GB/s with **no copy, no registration, no fill, no eviction, no per-op remap** — a strict superset of what a
> window offers — and it measures **8.5 t/s against 8.0 (+6 %)**.  The reason is that the 122B is 7.4 GiB larger
> than this box, and pinning `f` takes `f` out of the page cache at the same time, so the share streaming from NVMe
> **rises** with `f` (6.4 % -> 8.1 % -> 10.7 %).  A window changes *who reads*, never *how many bytes need reading*;
> it pays only where reads, not capacity, are the limit — which for this model on this box they are not.
>
> **The stop rule is not triggered, but §3's arithmetic is.**  No measured arm overlapped the CPU part with
> the GPU part, so the honest model is **additive**, and on a 123 GiB box the window size is bounded by what
> can be **pinned** out of the same 123 GiB the page cache needs.  The 2–3x target on the 122B is **not
> reachable**; a realistic target is **+10–40 % at a 20–40 % window** (+43 % predicted at 40 %).  The campain's
> aim becomes: *the pinned fraction runs at device speed, the window never leaves a table unserved*, and a
> static window's real win over the shipped tier is that it removes the 1 000–17 000 fills per 96 tokens the
> LRU does today.  **The read-band question is still open** (the 35B is latency-bound at 46 GB/s of effective
> expert traffic and cannot resolve 60 vs 230 GB/s) — see §9.4.

Cut against the **r243 tree** (`release.json`: tip `72f7753f`, tree `e9300f1`).  Every file this campaign cites is
identical to r242's (r243 touched only `src/models/qwen4exp.cpp`).

---

## 1. The problem, in one paragraph

On a shared host/device pool (a UMA APU) "VRAM" and host RAM are the **same DRAM**: the page cache holding the
model mapping is already resident, and the GPU reading a resident buffer measures **194–199 GB/s** while 15 CPU
cores read the same bytes in place at **~41 GB/s effective**.  A model whose expert set does not fit therefore has
exactly two usable behaviours today: *all* experts on the device (impossible — the set is bigger than RAM) or the
**CPU expert path** (`-ncmoe`; P0 measures **5.7–8.0 t/s** steady state on Qwen3.5-122B-A10B Q8_0 here — see the
correction below).  The shipped discrete tier (arena + pinned pool + copies) was recorded as **3x slower** than
that CPU path at this size class, because on UMA its copy buys no faster medium and its per-op remap costs a host
round-trip per routed op.

**The goal is therefore not residency** — the kernel's page cache already holds the bytes — **but the GPU reading
the hottest experts where they are: no copy, no eviction, no per-op host round-trip.**

## 2. The design

One tier, chosen once, read in place:

1. **A static window.**  At load, the persisted **expert heat prior** picks the hottest experts of each table; their
   bytes are copied **once** from the model mapping into an anonymous, 4 KiB-aligned buffer (the "window"), which is
   then registered `ReadOnly` for the GPU.  **Fill first, register after** — measured: register-then-fill is
   70–210 MB/s (page-fault bound) against ~7 GB/s the other way round, and 193 GB/s `pread` / 6.5 GB/s `O_DIRECT`
   into an already-registered range.
2. **Static residency ⇒ static remap.**  Because the window never changes, each table's `expert → slot` map is
   published **once at load**.  The per-op routing readback (`takeover`/`get_ok` in the old census, one synchronise
   per routed op, 144/token on the 122B) disappears by construction — that is the single biggest cost this design
   removes, and it needs no new sync-free machinery.
3. **The complement is not the tier's problem.**  Experts outside the window keep the CPU expert path (they are in
   page cache already, no copy).  The split is **per expert**, which is what the heat prior ranks.
4. **No arena, no eviction, no demote, no slab arena.**  The slab keeps its one remaining job — owning the compute
   work region — and nothing else.
5. **User surface (maintainer decision, 2026-11-10):** on a UMA device `-lme N` is the window size **in GiB**
   (`auto` = the policy share); `-ncmoe`/`-cmoe` are a no-op with one WARN (the bytes are in the same memory, so
   they only ever chose *who computes*).  If the model **fits** the device share, the tier is not engaged at all
   (advisory only).  If it does not fit, the run **declares it loudly** — heat-map mode, window size, overflows
   read from the page cache / storage.

## 3. Why this should be faster (the arithmetic to beat)

Per token the 122B routes ~3.8 GiB of expert bytes.

| read path | rate | ceiling |
|---|---|---|
| CPU, in place from page cache (today) | ~41 GB/s | **5.7–8.0 t/s** (P0, §P0 4 — state-dependent) |
| GPU, in place from a registered window | ~194 GB/s | **51 t/s** |
| mixed, window fraction *f* | `1 / (f/194 + (1−f)/41)` | f = 0.6 → ~77 GB/s → **~20 t/s** |
| mixed, if CPU and GPU overlap | `194f + 41(1−f)` | f = 0.6 → ~133 GB/s → **~35 t/s** |

So the naive target was **2–3x the CPU path on the 122B** with a 60 % window, and the *split* is the heat map's
job.  **P0 measured that split and refuted the overlap row**: the CPU part adds its time, it does not overlap
([`p0/RESULTS.md`](p0/RESULTS.md) §4).  Use the additive row instead, and remember that window bytes are *pinned*
bytes taken away from the page cache the CPU complement reads from.  Any result below the **CPU path's own**
measured rate (5.7–8.0 t/s on the 122B here) means the per-table split or the in-place read is not working, not
that the box is slow.

> **CORRECTION (2026-11-10): the "10.58 / 10.90 / 10.89 t/s" 122B CPU-path figure has no arm behind it.**  It
> appears only in `archive/work/apu-expert-pool/CONCLUSION.md`, while that campaign's own
> `MEASUREMENTS.md` has no 122B section at all, its `README.md` §11 says "the **>RAM acceptance arm has not been
> run**", and its `HANDOVER.md` opens with "the case the campaign exists for — a model LARGER than RAM — has not
> been run".  **P0's own 122B arms are the first recorded ones on this box**: `-ncmoe 48 -c 8192`, cache off,
> marginal **8.0 t/s** over tokens 64..320 and **5.7 t/s** over 320..600 — the rate *falls* with length, because a
> 123.4 GiB set cannot stay in a 123 GiB box's page cache.  Treat the old figure as unverified; use the measured
> range.  (The ratio matters: the old number made the CPU path look 25–90 % faster than it is, which made the
> campaign's upside look proportionally larger.)

## 4. The one question that decides whether this is a campaign — **ANSWERED (P0, 2026-11-10)**

> Measured, on `halo`, Qwen3.6-35B-A3B Q4_K_M: the in-place pool read is **+71 % over the CPU expert path and
> −4.5 % against device memory**; one routed op **can** mix window-resident and pool-cold experts (free); a
> **table** without a GPU representation costs the entire win.  Full arms, censuses and the corrected arithmetic:
> [`p0/RESULTS.md`](p0/RESULTS.md).  The subsections below are the original questions, kept as asked.

### The questions as posed

**Can a table op run with *some* experts in the window (GPU, in place) and the rest on the CPU, cheaply?**

The delivery already encodes expert location in the id (`n_res_slots + e` = the "cold" region read in place from
the host alias, `MOE_HOST_POOL_COLD`), and it was exercised at 2.6 % of accesses in the old campaign — but never as
100 % of a table's read path, and the repo notes that a table with no arena "stands the cache-band MoE fusions
down (the per-op path), ~8x slower" **on a discrete box over PCIe**.  That number may not transfer to UMA.

P0 must answer it with measurement before any production code:

- **P0.1** — on Qwen3.6-35B-A3B (where the arena tier works at full-slot residency: 59-61 t/s on this box), force *every* expert to
  be served in place from the pool (small arena so nothing is resident, full pool, `MOE_HOST_POOL_COLD`) and
  measure decode against the CPU path.  If 100 % in-place pool is at least CPU-parity, the shape is viable.
- **P0.2** — the mixed case: a window covering ~50 % of a table, the rest to the CPU.  Measure the per-op cost of
  the split (is it two routed ops, or one op with mixed ids?).  **This is the risk**: if ggml cannot express one
  routed op with mixed locations, the design needs a split op per layer and must account for that.
- **P0.3** — the static remap: confirm the id→slot map can be uploaded once at load and never touched (no
  per-token readback) and that a captured graph replays correctly across a decode.

**Stop rule:** if P0.1 or P0.2 shows the in-place/mixed path is worse than the CPU path at 100 % residency, stop
and write the negative result — that is a successful campaign outcome, and cheaper than the code.

## 5. The plan after P0

| step | what | accept |
|---|---|---|
| **P1 (first)** | **the ballast rig** (`wip/bench-56gb/`, IQ4_NL, 31 GiB ballast): static window vs the copy tier vs the CPU path — the only configuration on this box with slack for a window (copy tier 44.0, CPU path 31.0) | the window is ≥ the copy tier (44.0 t/s) with no per-miss copy and no evictions; text byte-identical; the same window sizing must not regress the discrete box |
| P1 | window sizing and selection: `-lme N` GiB, heat-prior ranking, fill-then-register, one WARN line declaring heat-map mode and the window size | the window is exactly N GiB, registered RO, and the census reports its coverage |
| P2 | static remap published at load; the window's tables route to it with no per-op readback | **the ballast rig's tier arm > 44.0 t/s** (the 122B is a capacity wall — the old "> 10.9 t/s" criterion is unreachable here, P0 addendum §C); no `takeover` per token in the census |
| P2b | **all-or-nothing per table**: window sizing must never leave a host table without an arena/window slot, and the census must assert `tables == n_host_tables` (P0: losing 20 % of the tables costs the whole win) | no arm with a partial table set; a `tables` shortfall is a hard failure, not a WARN |
| P3 | the complement: **GPU-served in place** where it can be pinned, CPU only where it cannot, and the table is the granularity (the P0.2 answer) | the split is visible in the census and the sum of both paths is the measured throughput (additive — do not budget for overlap) |
| P4 | policy: fits → no tier (advisory); does not fit → heat-map mode declared; `-ncmoe` no-op WARN; kill switches | `ENVIRONMENT.md` rows, defaults per the default-on policy, no change on a discrete box |

Reuse, do not rewrite: the pool's per-table slots, `src_path`/`src_offs` registration from the loader, the
`pread`-straight-into-the-slot fill, `pool_mmap_thp_pin`'s fill-then-register ordering, the persisted heat prior and
its ranking, prepopulate and the exit census.  All of that is shipped and exercised on discrete boxes.  What must
**not** be carried over: arena sizing, table allocation, eviction, the sink/demote exchange, repack, and the
`n_res_slots` arena remap.

The interface is already narrow — one call decides what the expert op reads
(`ggml-cuda.cu:6015`, `moe_cache_get_table(dst, src0, …, &arena, &n_slots, &eb, &remap, &nu, &nt)`) — so this
campaign changes an accessor and a policy, not the op plumbing.

## 6. Gates (inherited; none may be skipped)

- **Text identity**: same-seed greedy text with **no `-lv`**, on `llama-cli`, against the best fitting arm; the
  arm's own `-lv 4` census is a **separate run**.
- **The rig**: the 31 GiB-ballast 56 GiB-emulation (`wip/bench-56gb/`, `ballast.c` mlock 31 GiB) is the
  campaign's **primary** rig after P0 (§P0 4): it is the only configuration on this box with slack for a window
  (IQ4_NL, copy tier 44.0 vs CPU path 31.0).  The 122B (`/llm/models/Qwen3.5/122B-A10B/Q8_0/`, 123.4 GiB, **no MTP drafter**) as
  the >RAM stress model; IQ4_NL (87 GiB, A3B, MTP drafter) as the control that must not regress (**44.0 t/s** with
  the old arena tier is the number to beat or match).
- **`check-gpu-faults.sh`** on every arm, plus the kernel journal.
- **No regression on a discrete box** at equal config (the shipped tier stays exactly as it is there).
- **No benchmark in parallel**; `HIP_VISIBLE_DEVICES` pinned; `-t` sized to leave the GPU IRQ cores free.

## 7. Landmines (each cost real time in the previous campaign)

1. **The slab needs work + floor + ring before it will arm** (~9 GiB for a 122B).  Below that the cache prints
   `STREAMING the experts from the host (stock -ncmoe behaviour)` and the run is the **CPU path** — three arms were
   mistaken for tier arms this way.  Check `table_pool_locked` / the census, never the t/s alone.
2. **Zero arena registers no tables at all** (the whole GPU-side cache stack is arena-centric).  If this campaign
   needs an arena-free path, that is exactly the new machinery, not a configuration.
3. **Fill before you register.**  Register-then-fill is 70–210 MB/s; fill-then-register is fast and a later
   *in-place rewrite* of a registered range is fine (10-coherence) — but a **miss** into a registered slot is a
   rewrite, so per-slot `Unregister → fill → Register` (~15 µs per 3.3 MB) is the form an evicting tier would need.
   A static window avoids the question entirely — that is part of why it is static.
4. **The exit census does not reliably print on `SIGTERM`** (1 of 6 server arms).  Use `llama-cli` for a census, or
   add a mid-run report before relying on it.
5. **Do not size anything from "free VRAM"** on a shared pool; use `MemAvailable` taken **once**, and remember the
   weights/KV/compute are already in it.
6. **`--lazy-mode auto` needs the declared load mode** (`-lm mmap`), which is now fixed in the fork; an APU with
   `-lm auto` still resolves to `none` (maintainer decision: an APU declares its load mode).
7. **`llama-server` + `-lv 4`**: text comes back over HTTP so the log cannot pollute it; for `llama-cli` the text
   arm is **no `-lv`** and `-lv 3` is the trap.
8. **A dead server must not hang a measurement.**  Poll health in the shell with a bounded timeout *and* watch the
   process; give every request its own `timeout`; run the workload in its own session so one `SIGKILL` to the group
   ends it.  (Two runs in the previous campaign sat for 10–24 minutes on a server that had already died.)

## 8. What is inherited in the fork

Branch **`wip/apu-expert-pool`** (do not build on it blindly):
- `70e0eda4a` — the four validated fixes (arena allowance net of device use + on-demand mapping; pool ceiling keyed
  on the effective percentage; one snapshot + share for the advisory; lazy AUTO vs the declared load mode).  These
  are **real defects of the shipped path** and are wanted regardless of this campaign's outcome.
- a later commit — the retired experiment (pool-only mode, `g_arena` split, `-lme`-in-GiB plumbing, the heat-map
  declaration, the RO guard).  Take the small reusable parts (the `-lme` GiB semantics and the declaration) and
  leave the pool-only mode behind; it is proven inert.

Start this campaign's branch from `release.json.base` + `scripts/apply-all.sh` + the `70e0eda4a` diff, or cherry-pick
it — not from the retired experiment.

## 9. Open questions for the maintainer

1. Window sizing policy: a fixed `-lme N`, or an auto share that leaves the OS and the page cache their room?
   (The previous campaign's 75 %/10 %/15 % split assumed an arena; a window wants its own rule.)
2. When the window cannot hold the hot set, is a *slow* refresh acceptable (re-pick the window between sessions from
   the persisted prior) or must it be fixed for the life of the process?  (Fixed is simpler and is the assumption
   above.)
3. Does the discrete path keep the arena **and** the pool as shipped (assumed yes — nothing here touches it)?
4. **ANSWERED (P0 addendum §A): the window's effective bandwidth, per launch shape.**  A registered
   read-only window measures **231.0 GB/s against `hipMalloc`'s 232.1** (identical at 8/32/128/320 rows), and
   the band is set by the **launch shape**: 8 routed rows per launch = 95–103 GB/s (177 with a 1024-thread
   block), ≥32 rows = 205–232.  So use **~100 GB/s for a per-layer decode op** in §3's table, not 194, and
   treat the rows-per-launch × block-size product as a P1 lever.  `p0/RESULTS.md` §A has the table.
   What is still open is **capacity on the target box**, not bandwidth: see §P0 4 — the 122B has no slack,
   so the campaign's home is the capacity-constrained rig.
5. **NEW (P0): is a window that is *not* a copy cheaper?**  Registering a `/llm/models` file mapping
   read-only measured 230.5 GB/s too (`archive/work/apu-expert-pool/MEASUREMENTS.md` §3), and the pages are
   already in the page cache and shared with the CPU path — i.e. a *zero-copy* window, with no anonymous
   buffer and no fill.  The catch is the same one that bounds everything else: registration **pins**
   (probe §9), so the window is still spent out of the page cache's own budget.  Worth a P1 arm.
