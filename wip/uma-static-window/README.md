# wip/uma-static-window — a heat-ranked, read-only **static window** for host-resident MoE experts on a UMA APU

**Status: OPEN — P0 DONE, P2 DONE (the window is correct and free), and the campaign's remaining arms are
BLOCKED BY A CODE CONSTRAINT rather than by a measurement (2026-11-10, finding 7).  Still nothing in `patches/`.**
This campaign exists because the previous one
([`archive/work/apu-expert-pool/`](../../archive/work/apu-expert-pool/CONCLUSION.md)) disproved its own thesis.  Read
the CONCLUSION first; this file assumes it.

> ### P2 verdict (2026-11-10 — the full record: [`p0/RESULTS.md`](p0/RESULTS.md) §L and §M)
>
> **7. The zero-copy window is CORRECT, and a PARTIAL window WORKS — the campaign's headline case is measured.**
> Two layout defects (not the file-offset arithmetic §K suspected) produced the wrong data, both invisible to a
> bandwidth probe: the kernel indexes the cold stride in **type blocks**, so a pitch of `host_bytes + page`
> truncated 64 B/slot and drifted linearly; and a pitch of `host_bytes` makes consecutive slots **share a page**,
> coherent only when the next slot's expert is file-adjacent — never for a heat ranking.  The pitch is now the
> smallest multiple of `lcm(page, ggml_type_size)` that holds the expert plus its residue (35B 589824 -> 626688,
> 122B 3342336 -> 3411968), registered per slot, refused when a stride is not a page multiple.  35B `-ncmoe 40`,
> `-n 96 seed 42`: **377 chars `sha=ae39ec41bc1e`** (the reference) at **59.2 t/s**, heat-ranked, `fills=0`,
> **cold 99.61 %**, for ~0 memory; a second prompt matches byte for byte.  **Partial coverage needed three fixes,
> none of them new kernel machinery**: a window-served expert must never be *admitted* into the arena (it is already
> one load away, and it competes for the very slots the complement needs); a window must be exempt from the arena's
> complement *re-layout*, which was silently freeing the hottest 8 slots per table and could never refill them; and
> the device budget must cover the work region + the H2D-ring hole + the arena.  With those, the 122B runs at
> **13.5 t/s (88 GiB window) / 15.0 t/s (96 GiB)** against the CPU path's **8.0** — **+69 % / +88 %**, text
> byte-identical to the copy-pool reference (`sha=58c95d8789b0`), `decline_all` down from `9360/9360` to a handful.
>
> **8. And the memory argument that killed the 122B was a myth: a registered window is NOT pinned.**  `Unevictable`
> and `Mlocked` stay **0** for the whole run, `Cached` grows to ~100 GiB and `MemAvailable` only dips ~17 GiB for an
> 88 GiB window: a window over `MAP_SHARED|PROT_READ` file pages is a *device mapping over the page cache*, not
> locked RAM.  Its pages are reclaimable and shared with the CPU path, so its cost is page cache and its size is not
> bounded by a pin budget — which is why 88-96 GiB windows run with no swap.  §H's "96 GiB + 8.6 GiB of weights does
> not fit" was the wrong conclusion from the right measurement (§I.1's "MemAvailable moved 127 MiB for 24 GiB" is
> the same fact).  The remaining gap against the prior's own 97.2 %-of-mass prediction (88-93 % measured) is
> **prior transfer** across prompts, not ranking.

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

1. **A static window.**  At load, the persisted **expert heat prior** picks the hottest experts of each table; the
   window is **the model mapping's own pages for those experts, registered `ReadOnly` in place** — NOT a copy.
   *(Amended 2026-11-10, P0 §H: the copy this step used to specify is exactly what makes a 96 GiB window 96 GiB
   **more** than the box has — measured, it swapped 22 GiB and decoded at 0.8 t/s.  Registering the existing pages
   costs no memory, and `p0/band.cpp`'s `file-ro` arm measures the result at **221.0 GB/s at 32 rows/op and
   99.2 GB/s at 8** — identical to `hipMalloc`.  Registration pins, so the window is sized to fit; the ranges are
   per expert (3.19 MiB each on the 122B), and the complement stays pageable.)*  Fill-then-register survives only as
   the fallback for a range that is not already resident — register-then-fill is 70–210 MB/s (page-fault bound)
   against ~7 GB/s the other way round.
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
