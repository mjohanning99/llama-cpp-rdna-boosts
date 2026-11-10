# P0 arm index

Every arm is one `llama-cli` load; `./arms.sh <arm> [n]` re-runs it (default `-n 96`).  The verdict and the numbers
are in [`RESULTS.md`](RESULTS.md); this file is only the map from an arm name to what it configures.

| arm | purpose |
|---|---|
| `ref-device` | `-ncmoe 0`: every expert on the device — the ceiling, and the text reference |
| `cpu-path` | `-ncmoe 40 --host-experts pool -lm mmap`, cache off — the CPU expert path (the number to beat) |
| `cold100-strict` | all tables registered, **1 slot/table**, admission never admits, covering **read-only** pool: P0.1 |
| `cold100-strict-wr` | the same with the pool registered **writable**: the registration A/B |
| `window5/25/50/75` | a heat-ranked per-expert window of that % of each table + a cold complement, all GPU-served: P0.2 |
| `window100` | `SLOTS=256`: the arena runs 0.4 GiB short, 24 tables fall to the CPU — the cliff |
| `cold100` | a 6 GiB slab cap: the slab arms too small, 39 tables fall to the CPU |
| `pinned-set` | `--host-experts pinned` on an iGPU (refused — the r243 policy) |

## Three traps this harness had to climb

Each one is preserved in `results/` as the failed arm that hit it.

1. **A slab budget below `used + reserve + 2 chunks` declines**, and the run silently becomes the CPU path
   (`the movable-boundary slab … could not be created` → `STREAMING the experts from the host`).  The t/s alone
   cannot tell you: the arm measures exactly the CPU path's rate.  Check the census.
2. **`GGML_CUDA_SLAB_MIN_ARENA_MIB` (default 2048) is a hard floor**, so a deliberately *small* arena needs it
   lowered — otherwise the same silent fallback.
3. **On a shared pool `MOE_EXPERT_CACHE_MIB` is the total device budget, not the arena size** (the slab is capped
   to it, and the arena gets `cap − used − reserve`, where `used` is what the model already holds on the device —
   3.33 GiB here, because every expert is host-resident).  That is why the small-arena arms also set
   `GGML_CUDA_SLAB_RESERVE_MIB`.

And the one that is not a trap but a design constraint: **a table that gets no arena slot is not taken over at all**,
so its layer's MoE runs on the CPU and the whole decode falls to CPU parity (`window100`, `cold100`).  A partial
table set is a hard failure, not a slow arm.
