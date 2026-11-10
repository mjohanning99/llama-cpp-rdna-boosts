# NEXT-SESSION-PROMPT — `wip/uma-static-window`

Hand this verbatim to a cold session.  It assumes no memory of any previous session.

---

You are continuing the `wip/uma-static-window` campaign on this box (`halo` — Ryzen AI MAX+ 395, **gfx1151** UMA APU,
123 GiB, kernel 7.2.13, ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`, 16 cores).  It is **not** a fresh scoping exercise:
P0, P1, the design amendment and P2 steps 1-2 are done and measured, and there is exactly **one known bug** standing
between the design and its first honest measurement.

## Read first, in this order

1. `~/llama-cpp-rdna-boosts/AGENTS.md` (policy: apply/push/gates/default-on/report-before-commit).
2. `~/llama-cpp-rdna-boosts/wip/uma-static-window/README.md` — the design as **amended**, and the P0/P1 verdict block
   at the top (six numbered findings, including the amendment: the window is **the model's own pages registered
   read-only**, NOT a copy).
3. `~/llama-cpp-rdna-boosts/wip/uma-static-window/p0/RESULTS.md` — all of it, but especially **§H** (the arithmetic)
   and **§K** (the open bug).  §G is superseded, §K is current.
4. `~/llama-cpp-rdna-boosts/archive/work/apu-expert-pool/CONCLUSION.md` — why the arena thesis died (its 122B
   numbers are CORRECTION-flagged as having no arm behind them; do not quote them).

## State

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`apu-expert-pool`** — commit and `git push origin apu-expert-pool` when you change it |
| fork | `~/llama.cpp`, branch **`wip/uma-static-window`**, tip **`437dd6491`** (r243 delivery + the `apu-expert-pool` enablement/fixes + the P0 RO hunk + the P2 window).  Never push upstream. |
| build | `cd ~/llama.cpp && cmake --build build-wip -j 16` (or `BUILD_DIR=build-wip ~/bin/build-llama-rocm`).  **Never** touch `build-rocm`.  Confirm with `nm -D`. |
| delivery set | **nothing in `patches/`; `release.json` untouched; no default changed.** Everything is behind opt-in env values. |

## What is established (measured, with the harness in `p0/`)

**The transport is free.** On this box `hipMalloc` is GTT — the same DRAM as host RAM — and a registered read-only
window reads **231.0 GB/s against hipMalloc's 232.1** (identical at 8/32/128/320 routed rows).  The band is set by the
**launch shape**: 8 routed rows/launch = 95–103 GB/s (177 with a 1024-thread block), ≥32 rows = 205–234.
`p0/band.cpp`, RESULTS §A.

**100 % in-place service = device parity.** 99.48 % of routed accesses served in place from a read-only pool with one
arena slot per table: **59.8-60.1 t/s** vs **62.7-62.9** all-device and **35.0-35.1** CPU path (35B, `-n 96`).
`p0/arms.sh cold100-strict`, RESULTS §P0 1.

**The split granularity is the TABLE, and it is a cliff.** Losing 20 % of the tables costs the whole win (96/120
tables GPU-served → 35.5; CPU path → 35.0).  Never size a window so a host table gets no GPU representation;
assert `tables == n_host_tables`.  RESULTS §4.

**The heat arithmetic supports the maintainer's model.** From the persisted `MHT2` prior (`p0/prior.py`), 122B Q8_0:
48 GiB → 85.89 % hit, 80 GiB → **98.42 %**, **96 GiB → 99.94 %** (2 MiB/token of misses), 104 GiB → 100 %.  The mass
per table is **uniform to 1 %**, so `-ncmoe` (layer placement) covers only its byte fraction — which is why
`-ncmoe 24` (57.4 GiB on the device) measures 8.5 t/s against the all-host 8.0.  RESULTS §H, §G(superseded).

**A zero-copy window is buildable and free.** `p0/winprobe.cpp`: 7700 heat-ranked slots × 3 342 336 B = 23.97 GiB of
the real shard, `MAP_FIXED` per expert into a compacted VA range, one `hipHostRegister(ReadOnly)`:
fault-in 4.99 s at 4.9 GB/s, **register 0.45 s at 54 GB/s**, **MemAvailable moved 127 MiB for 24 GiB** (no copy),
read **234.2 GB/s wide / 199.0 GB/s at 8 rows with 1024 threads**.  RESULTS §I.1.

**The P2 window mode exists and runs** (`MOE_HOST_POOL_ALLOC=window`, fork tip): 35B, 18.6 GiB of heat-ranked
residency, `fills=0 evictions=0`, **87.8 % cold in place, 58.5 t/s, ~0 memory** (the copy-pool arm needs the
18.6 GiB for 59.8-60.1).  RESULTS §K.

**Baselines to beat** (all on this box, `-t 12`, `HIP_VISIBLE_DEVICES=0`, no MTP):

| case | number |
|---|---|
| 35B `-ncmoe 0` (all device) | 62.7 t/s · text `ae39ec41bc1e` at `-n 96, seed 42, temp 0` |
| 35B CPU expert path (`-ncmoe 40`, cache off) | 35.0 t/s · same text |
| 35B copy-pool 99.5 % cold | 59.8-60.1 t/s · same text |
| 122B all-host CPU path | **8.0 t/s** marginal (tokens 64..320), 5.7 (320..600) |
| 122B with 57.4 GiB of experts on the device | 8.5 t/s |
| 122B prefill, 4059 tokens, `-ub 4096`, cache armed | **175.2 t/s** (23.17 s), of which **9.98 s is the fill memcpy path** (33.15 GiB at 3772 MiB/s) |

## TASK 1 (do this first): the window's data is wrong — fix it

The mode runs but the bytes at a slot are not that expert's.  35B, `-n 96`:

```
reference (all-device / CPU path / copy pool): 377 chars  sha=ae39ec41bc1e
window arm, heat-ranked slot order:            114 chars  sha=4c3e25f4ca2b
window arm, MOE_HOST_POOL_WINDOW_ORDER=id:     114 chars  sha=4c3e25f4ca2b   <- same, so NOT the id convention
```

**The naming measurement** (do this before changing anything): for one table, read `host_bytes` from window slot `s`
and compare with `t.host + rank[s]*host_bytes` (the pageable mapping).  A mismatch localises the file-offset
arithmetic.  The suspect, from `pool_window_map_locked`:

* it maps from `p.file_offs + e*host_bytes`, where `p.file_offs = t.src_offs` — the pool's *other* users treat
  `t.src_offs` as the **slice** offset (`moe_cache_get_cold` adds it to `base_dev`), so if it is not the tensor's
  file data offset the mapping starts in the wrong place;
* the loader's real `(path, data-offset)` pair is what is needed (`llama_model_loader` knows it — see how
  `t.src_path`/`src_known` are filled at registration, and `MOE_HOST_POOL_BULK`'s mirror fill, which relies on
  `file_offs + e*host_bytes` being the GGUF offset of expert `e`);
* also verify `win_residue` (`file_offs % 4096`) and that `cold_base = base_dev + win_residue` with
  `win_stride = host_bytes + 4096` is what the kernel actually consumes (it reads
  `cold_base + slot*win_stride`, row stride from `cold_row_bytes`).

**Acceptance:** the window arm reproduces **`sha=ae39ec41bc1e`** (377 chars) at `-n 96, seed 42, temp 0, no `-lv``,
and keeps `fills=0` with ≥87 % cold.  If it also holds on a second prompt (use one from `~/llama-cpp-rdna-boosts/prompts/`),
the mechanism is done and can be measured.

Working config that reaches the mode (copy it, do not rediscover it):

```bash
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf
env MOE_HOST_POOL_ALLOC=window MOE_HOST_POOL_FRAC=100 MOE_HOST_POOL_SINK=0 MOE_HOST_POOL_RAM_PCT=85 \
    MOE_EXPERT_CACHE_MIB=24576 GGML_CUDA_SLAB_RESERVE_MIB=1024 \
    MOE_EXPERT_CACHE_SLOTS=1 MOE_EXPERT_CACHE_TOUCH=1000000 \
  ~/llama.cpp/build-wip/bin/llama-cli -m $M -ngl 99 -c 4096 -p "The capital of France is" -n 96 \
    --seed 42 --temp 0 --single-turn --no-display-prompt -t 12 -ncmoe 40 --host-experts pool -lm mmap
```

## TASK 2: the arms that decide it (only after TASK 1 passes)

1. **122B decode at 96 / 88 / 80 GiB window.**  Prediction (RESULTS §H): ~26 t/s at 99 GB/s, ~48 at 199 GB/s, against
   8.0 today.  Memory: window + 8.6 GiB of weights ≈ 105 GiB of 116 — feasible, but **the pool ceiling must be
   raised deliberately** (`MOE_HOST_POOL_RAM_PCT`) and any arm that pins >90 GiB must be treated as a swap-containment
   test, not a throughput arm (see the trap below).  Bound every run (`timeout -k 5`), watch `MemAvailable`/swap in a
   sampler, and never leave it unsupervised.
2. **122B prefill, 4059 tokens, `-b 4096 -ub 4096`** against **175.2 t/s**: 9.98 s of the 23.17 s is the fill memcpy
   (33.15 GiB at 3772 MiB/s).  A window makes that read in place at 199-234 GB/s.  Predicted ~300 t/s.  The ~13 s of
   compute is the floor (1.9 TFLOPS effective); a 700 t/s-class number would need ~8.6 TFLOPS and is not reachable
   here.
3. **Then** the campaign's remaining question: does a static window beat the *shipped* tier at equal bytes
   (`-lme N` copy pool + arena)?  On the capacity rig (`wip/bench-56gb/`) the shipped tier measures **26.3 t/s**
   against the CPU path's 14.8 — that is the number a window must beat there.

## Open side-findings worth reporting, not fixing blind

* **`-ncmoe` + cache off + a prompt whose ubatch reaches the staging path (`min_tokens=64`) can stall outright** on
  this box: 201 tokens → no completion in 8.6 min; 394 tokens → 12 min, stuck in `poll`, 109 GB RSS, box at 3 GiB
  free.  With the cache armed the same prompt finishes at **10.1 t/s with 191 MiB of expert H2D traffic**.  Confound:
  the two arms differed in page-cache state (cache-off ran cold).  The clean test (cache off, warm cache, 201 tokens)
  has not been run.  RESULTS §I.2.
* **This branch's UMA pool ceiling defaults to 10 % of `MemAvailable`** (11.97 GiB) and silently scales a planned
  96 GiB pool by 0.122 (26 slots/table, h = 3.8 %, 82 170 evictions).  Any "the tier can't hold N" result must first
  confirm the clamp did not fire.
* `MOE_EXPERT_CACHE_MIB` on a shared pool is the **total device budget**, not the arena size (`arena ≈ cap − used −
  reserve`; `used` was 3.33 GiB on the 35B because every expert is host-resident).

## Traps (each cost real time this session)

1. **A slab budget below `used + reserve + 2 chunks` declines and the run silently becomes the CPU path**
   (`STREAMING the experts from the host`).  The t/s is then exactly the CPU path's — check the census, never the rate.
2. **`GGML_CUDA_SLAB_MIN_ARENA_MIB` (default 2048) is a hard floor**; a deliberately small arena needs it lowered.
   `GGML_CUDA_SLAB_RING_MIB` below ~6529 MiB warns and falls back for this model class.
3. **Raising the pool ceiling pins the box into swap**: 96 GiB pinned + the mapping it was filled from = 22 GiB of
   swap and 0.8 t/s.  A window avoids this by construction; a *copy* pool cannot.
4. **Text arms take no `-lv`; censuses are a separate `-lv 4` run** (`-lv 3` is the trap).  `HIP_VISIBLE_DEVICES=0`
   on every arm.  The CPU path and the GPU tier legitimately differ in text (different kernels) — only compare text
   within a kernel path.
5. **Never leave a workload unsupervised**: own session, `timeout -k 5`, watch the process, kill the group.  One arm
   this session had to be killed after 12 minutes stuck in `poll`.
6. Run `~/llama-cpp-rdna-boosts/scripts/check-gpu-faults.sh` on every arm.

## Harness (all in `wip/uma-static-window/p0/`, logs in `results/`)

| tool | what |
|---|---|
| `arms.sh <arm> [n]` | 35B arms: `ref-device`, `cpu-path`, `cold100-strict` (+`-wr`), `window5/25/50/75/100`, `cold100` |
| `long.sh <arm>` | the same arm at `-n 96` and `-n 512`, with the **marginal** rate over 96..512 |
| `arms122.sh <ncmoe> [n]` | 122B arms with a memory sampler |
| `p1.sh <arm> [n]` | capacity rig arms (ballast in its own session) |
| `cpuwatch.sh <arm>` | CPU % of one core + GPU busy % — proves whether the CPU is computing MoE |
| `band.cpp` | the window band probe (`dev`/`anon-ro`/`anon-rw`/`file-ro`, rows × block size) |
| `winprobe.cpp` | the zero-copy window builder (VA + `MAP_FIXED` + register) and its band |
| `prior.py` | the achievable hit rate from the `MHT2` sidecar at any resident budget |

Commit and push every delivery-repo change to `origin apu-expert-pool`; report findings to the maintainer before
changing any default, and do not put anything in `patches/` without a validated win on this box.
