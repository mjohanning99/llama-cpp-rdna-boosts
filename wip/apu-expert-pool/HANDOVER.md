# HANDOVER — `wip/apu-expert-pool` (cold-session continuation)

**Written 2026-11-09, end of the first development session.  Read this file top to bottom before touching anything.**
The companion [`NEXT-SESSION-PROMPT.md`](NEXT-SESSION-PROMPT.md) is the text to hand a fresh session.

Status in one line: **the tier works end to end on a UMA APU for a model that FITS, and the case the campaign
exists for — a model LARGER than RAM — has not been run.**

---

## 1. Where everything lives

| | |
|---|---|
| **delivery repo** | `~/llama-cpp-rdna-boosts`, branch **`apu-expert-pool`**, pushed to `origin` |
| delivery commits | `4846380c` scoping note → `b1786bc0` increment 1 → `db3e7c1d` mmap decision → `92955dd5` P2 → `ebc61277` budget policy (0003) → `90e021a4` single snapshot (0004) |
| **fork checkout** | `~/llama.cpp`, branch **`wip/apu-expert-pool`**, pushed to the personal fork |
| fork commits | `38829df20` increment 1 → `ad3a535a2` P2 → `8c9ad0d19` budget split (0003) → `4cc296f6e` single snapshot (0004) |
| base | r243: tip `72f7753f`, tree `e9300f1` (`release.json`) |
| patches | `wip/apu-expert-pool/0001-…` … `0004-uma-single-snapshot.patch`, each `git am`-verified against a fresh r243 worktree |
| build dir | `~/llama.cpp/build-wip` (deliberate: **do not** clobber `build-rocm`) |
| notes | `README.md` (gate map, measured foundation, traps, §10-§12 dev logs), `IMPLEMENTATION.md` (the phased plan), `MEASUREMENTS.md` (probe log + numbers), `probe/` (10 standalone HIP programs) |

Nothing is in `patches/`.  `release.json` is untouched.  No `TODO` was opened — the open decisions are in
`README.md` §7 and §11.

## 2. Build and run (exact, this box)

```bash
# build (FULL build, not a target-only build -- see landmine L1)
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
cmake --build build-wip -j 16

# sanity: the UMA hook must be defined in the library, not left undefined
nm -D build-wip/bin/libggml-hip.so.0 | grep uma_free        # want: T _Z28moe_cache_set_uma_free_bytesl
nm -D build-wip/bin/libggml-base.so | grep uma_free         # want: T ggml_backend_dev_moe_cache_set_uma_free
```

The working arm (**`-lm mmap` is mandatory**, see L3; `--host-experts pool` is deprecated in r243 — `-lme auto`
is the modern spelling and sets the same `AUTO` mode):

```bash
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0 timeout 1800 ./build-wip/bin/llama-cli -m $M -ngl 99 \
  -ncmoe 8 -lme auto -lm mmap \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4
```

Hash arms take **no `-lv`** (it pollutes the text); censuses take **`-lv 4`**.  Two runs, never one.

## 3. What is implemented (`0001`-`0004`)

| patch | what it does | kill switch |
|---|---|---|
| `0001` | loader: a shared pool refuses `pinned` but allows `pool`/`auto`; **refuses** the placement outright when no pool-backed master exists (never silently falls back to the pinned arm that faulted); `AUTO` + a host-expert placement keeps mmap on an iGPU | `MOE_HOST_EXPERTS_UMA=0`, `MOE_UMA_MMAP=0` |
| `0002` | `moe_cache_preflight` UMA arm: arena sized from RAM, not "free VRAM"; `ggml_cuda_slab_set_total_cap()` so the slab cannot map the machine; `moe_host_master_buft()` offers `HostMap` to an integrated device **only while the pool is armed** (`moe_cache_host_pool_enabled()`) | `MOE_EXPERT_CACHE_UMA=0` |
| `0003` | the UMA budget split (maintainer policy): VRAM 75 % / pool 10 % of free system memory, ~15 % left to the OS; a **loud advisory** when the model already fits in "VRAM" (informs, never overrides) | `MOE_UMA_VRAM_PCT`, `MOE_UMA_POOL_PCT` |
| `0004` | ONE `MemAvailable` snapshot for both shares (loader → `ggml_backend_dev_moe_cache_set_uma_free()` → cache), so 75 % + 10 % are shares of the same figure | — |

Also present from `0001`: the host pool registers its device alias **read-only** when no arena is live
(`MOE_HOST_POOL_RO`, 233 GB/s vs 60 GB/s measured) — **but see L5: on this box the arena IS live, so the pool is
still registered writable and the 233 GB/s win is not yet realised.**

## 4. What has been MEASURED (all on `halo`, gfx1151, 123 GiB, ROCm 10.1.0)

Qwen3.6-35B-A3B Q4_K_M (21 603 MiB, 40 layers, 24 expert tensors at `-ncmoe 8`, host set 3814 MiB):

| arm | result |
|---|---|
| `-ncmoe 0` (all device) | reference text |
| `-ncmoe 8 -lme auto -lm mmap` | **text byte-identical**, no GPU faults, `ROCm_HostMap` masters, cache `fills=6144 tables=24 arena=3814 MiB` |
| advisory | fires: `this model (21603.0 MiB) FITS in this APU's usable device memory (~89542.4 MiB = 75% of MemAvailable)` |
| arena budget | `89542 MiB/device` — **the same figure as the advisory** (that equality is the point of `0004`) |
| `MOE_HOST_EXPERTS_UMA=0` | placement refused, `load_mode=none` (today's behaviour verbatim) |
| `MOE_EXPERT_CACHE_UMA=0` | slab not armed (cache off) |

**The single most important measurement:** the pool placement does **not** fault on gfx1151.  The historical
`mul_mat_q<Q2_0,16>` fault is therefore not an architectural limit of the tier; it is what a
placement-resolution fallback produces (see gate #5 in `README.md` §1).

## 5. The five gates and where each stands

| # | gate | state |
|---|---|---|
| 1 | `set_host_experts_umem` blanket refusal | **ungated** for pool/auto (`0001`) |
| 2 | `moe_cache_preflight` shared-pool refusal | **inert on this box, and that is a finding**: `--fit` pre-fills `g_budget` from free VRAM before the preflight's shared-pool branch, so the branch never fired and the expert cache has been running on the APU all along.  `0002` now bounds its budget instead |
| 3 | `common/fit.cpp` reserve refusal | **still on** — now prints a message that is wrong ("the expert cache is not used there"; it is).  P3 |
| 4 | `caps.mmap_support = type != IGPU` | **left alone by design** (maintainer decision 2026-11-09): an APU declares its load mode; `AUTO` + host-pool keeps mmap via `MOE_UMA_MMAP` |
| 5 | `moe_host_master_buft()` NULL for integrated | **closed** by `0002`, gated on the pool being armed |

## 6. NEXT STEPS, in order, with acceptance criteria

### S1 — the >RAM arm (**the reason the campaign exists; nothing else matters until this runs**)

```
/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/  (2 shards, 74 GiB, + mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf)
```

Run `-ncmoe 24` and `-ncmoe 48`, `-lme auto -lm mmap`, `-c 32768`, and record:

* greedy text vs the best fitting arm (hash with **no `-lv`**), and `check-gpu-faults.sh` on the log;
* that the **pool 10 % ceiling binds** (this is the first config where the host set can exceed it);
* `SLAB_INIT` size ≤ the budget, and that `MemAvailable` after load is above the OS reserve;
* census: non-zero evictions, `arena` > 0, `takeover=0`;
* the advisory should **not** fire (the model does not fit) — if it does, the fit test is wrong.

If this arm faults, stop and attribute it by address-to-buffer (`INVARIANTS.md` r214 §2: never by kernel name).

### S2 — unlock the read-only pool (L5)

The pool is registered writable on this box because the arena is live and the arena→pool **demote** writes the
backing from the device (`cudaMemcpyDeviceToHost` into `pool->base`).  Implement P2.4 (skip the demote on UMA —
an eviction then only invalidates the table entry) and flip `pool_ro` to true on UMA.  Expected: 233 vs 60 GB/s
on every pooled host read.  Accept: text identical, `gate-host-pool.sh` green, and the 60→233 GB/s visible in a
direct read benchmark.

### S3 — `scripts/gate-uma-arena.sh` (spec: `IMPLEMENTATION.md` §7)

8 arms: text identity (no `-lv`), >RAM, cold page cache, budget, both kill switches, the Windows carve-out arm,
`GGML_FORCE_NO_INTEGRATED=1`, and `MOE_HOST_POOL_FILL_FAULT=1` (must decline, never read a pageable master).
This is the promotion gate; nothing ships without it.

### S4 — P3 (the budget stops lying)

`common/fit.cpp`: with a UMA arena the reserve must be **taken from `MemAvailable`** instead of skipped as
"refused".  Today the fit pass reserves nothing and prints a message that is now false.

### S5 — housekeeping before promotion

`ENVIRONMENT.md` rows for `MOE_HOST_EXPERTS_UMA`, `MOE_UMA_MMAP`, `MOE_EXPERT_CACHE_UMA`, `MOE_UMA_VRAM_PCT`,
`MOE_UMA_POOL_PCT`, `MOE_HOST_POOL_RO`; `INVARIANTS.md` entries (register==pin; 233-vs-60 GB/s; the `DONTNEED`
silent zero; the anonymous-namespace rule); `MANIFESTS.md` gate row; a dated `WORKLOG.md` entry; the block-09/12
note in `patches/README.md`; `release.json` only via `scripts/make-release.sh`.

## 7. Landmines — every one of these has already cost time

* **L1 — a target-only build can leave a library with a stale undefined symbol.**  `cmake --build … --target
  llama-cli` reported success while `libggml-hip.so.0` kept `U moe_cache_set_uma_free_bytes`, and the binary then
  died at startup with `undefined symbol`.  Do a full `cmake --build build-wip -j 16` and check with `nm -D`.
* **L2 — a symbol added to `moe-expert-cache.cu` must be OUTSIDE the file's anonymous namespace** (it closes at
  `} // namespace`, ~line 5621).  Inside, it has internal linkage and the compiler drops it.  Cost: one full
  debug cycle.
* **L3 — `-lm mmap` is required for the pool tier.**  An APU advertises `caps.mmap_support = false`
  (`ggml-cuda.cu`), so `--load-mode auto` resolves to `none` and the pool's model mapping is never built.  `AUTO`
  + a host-expert placement keeps mmap (`MOE_UMA_MMAP`), but the maintainer's decision is that an APU declares
  its load mode — do **not** flip `caps.mmap_support`.
* **L4 — `g_budget` is pre-filled from free VRAM by `--fit`** (`g_auto = 0` **with** a free-VRAM figure,
  measured `74 489 620 070 B`).  Never test a budget for "explicitness" with `!g_auto`; key on the operator's
  `MOE_EXPERT_CACHE_MIB`.  This is also why gate #2 never fired.
* **L5 — the pool's read-only registration is tied to `!g_enabled`**, because the arena→pool demote is the one
  device-side writer of the pool backing.  With the arena live (this box) the pool is writable, i.e. the 60 GB/s
  arm.  Forcing `MOE_HOST_POOL_RO=1` with a live arena is refused loudly on purpose.
* **L6 — text arms take no `-lv`; censuses take `-lv 4`.**  `-lv 3` both pollutes the hash and hides the ggml
  censuses.
* **L7 — two log lines in the UMA preflight arm never reach the log** although the branch provably runs (the
  budget moves, the slab disarms).  Cosmetic, unexplained; **the census is authoritative**.  Do not write a gate
  that greps those lines.
* **L8 — `--host-experts pool` is deprecated in r243** (`common/arg.cpp` warns "use `--limit-host-experts`").
  Use `-lme auto`.  Both set `host_experts_mode = AUTO`, which is what the ungate keys on.
* **L9 — `HIP_VISIBLE_DEVICES=0`** on every arm.  One device on this box (the iGPU *is* the device), but pin it
  for determinism.
* **L10 — gates and benchmarks must not overlap** (`AGENTS.md`), and no benchmark in parallel with anything.

## 8. Decisions already taken (do not re-litigate)

1. **An APU declares `-lm mmap`**; `caps.mmap_support` stays false for every iGPU.  `AUTO` + host-pool keeps
   mmap (this is a load-mode special case, not a capability change).
2. **`--host-experts pinned` stays refused** on a shared pool until the fault is fully explained.
3. **UMA budget split: VRAM 75 % / pool 10 % / ~15 % to the OS**, all ceilings, all from **one** snapshot.
4. **The "this fits, drop `-ncmoe`" message informs and never overrides.**
5. A benign feature is **on by default** with a kill switch (`AGENTS.md`, default-on policy) — hence
   `MOE_EXPERT_CACHE_UMA` defaults on, and every switch is `=0` to disable.

## 9. File map of this campaign

| file | what |
|---|---|
| `README.md` | charter, the five gates, the measured foundation, the two shapes, the traps, §7 open questions, §10-§12 dev logs |
| `IMPLEMENTATION.md` | the phased plan (P0/P0.5/P1.0/P1.1/P1.2/P1.4/P2.0-P2.5/P3/P4), the gate spec, effort, rollback |
| `MEASUREMENTS.md` | the probe log: box, every number, reproduction commands |
| `probe/` | 10 standalone HIP programs (all build on ROCm 10.1), each with what it establishes |
| `0001`…`0004` | the development increments, `git am`-verified against r243 |
| `HANDOVER.md` | this file |
| `NEXT-SESSION-PROMPT.md` | the prompt to give the next session |
