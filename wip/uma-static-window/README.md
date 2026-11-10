# wip/uma-static-window — a heat-ranked, read-only **static window** for host-resident MoE experts on a UMA APU

**Status: OPEN — scoping note, nothing implemented, nothing in `patches/`.**  This campaign exists because the
previous one ([`archive/work/apu-expert-pool/`](../../archive/work/apu-expert-pool/CONCLUSION.md)) disproved its
own thesis.  Read the CONCLUSION first; this file assumes it.

Cut against the **r243 tree** (`release.json`: tip `72f7753f`, tree `e9300f1`).  Every file this campaign cites is
identical to r242's (r243 touched only `src/models/qwen4exp.cpp`).

---

## 1. The problem, in one paragraph

On a shared host/device pool (a UMA APU) "VRAM" and host RAM are the **same DRAM**: the page cache holding the
model mapping is already resident, and the GPU reading a resident buffer measures **194–199 GB/s** while 15 CPU
cores read the same bytes in place at **~41 GB/s effective**.  A model whose expert set does not fit therefore has
exactly two usable behaviours today: *all* experts on the device (impossible — the set is bigger than RAM) or the
**CPU expert path** (`-ncmoe`, 10.9 t/s on Qwen3.5-122B-A10B Q8_0).  The shipped discrete tier (arena + pinned
pool + copies) is **3x slower** than that CPU path at this size class, because on UMA its copy buys no faster
medium and its per-op remap costs a host round-trip per routed op.

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
| CPU, in place from page cache (today) | ~41 GB/s | **10.9 t/s** (measured) |
| GPU, in place from a registered window | ~194 GB/s | **51 t/s** |
| mixed, window fraction *f* | `1 / (f/194 + (1−f)/41)` | f = 0.6 → ~77 GB/s → **~20 t/s** |
| mixed, if CPU and GPU overlap | `194f + 41(1−f)` | f = 0.6 → ~133 GB/s → **~35 t/s** |

So the target is **2–3x the CPU path on the 122B** with a 60 % window, and the *split* is the heat map's job.  Any
result below 10.9 t/s means the per-expert split or the in-place read is not working, not that the box is slow.

## 4. The one question that decides whether this is a campaign (do this FIRST)

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
| P1 | window sizing and selection: `-lme N` GiB, heat-prior ranking, fill-then-register, one WARN line declaring heat-map mode and the window size | the window is exactly N GiB, registered RO, and the census reports its coverage |
| P2 | static remap published at load; the window's tables route to it with no per-op readback | 122B decode **> 10.9 t/s**; no `takeover` per token in the census |
| P3 | the complement on the CPU expert path, per expert (the P0.2 answer) | the split is visible in the census and the sum of both paths is the measured throughput |
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
- **The rig**: the 31 GiB-ballast 56 GiB-emulation (`wip/bench-56gb/`, `ballast.c` mlock 31 GiB) for the
  fits-vs-does-not-fit boundary; the 122B (`/llm/models/Qwen3.5/122B-A10B/Q8_0/`, 123.4 GiB, **no MTP drafter**) as
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
