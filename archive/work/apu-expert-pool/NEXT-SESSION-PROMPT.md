# OBSOLETE — do not hand this to a session

This prompt told the next session to run the >RAM arm of the **arena** design.  That design is closed:
the arena is a duplicate medium on UMA and the tier was measured 3x slower than the CPU expert path
(see [`CONCLUSION.md`](CONCLUSION.md)).  The live prompt is
[`wip/uma-static-window/NEXT-SESSION-PROMPT.md`](../../../wip/uma-static-window/NEXT-SESSION-PROMPT.md).

Kept only so the record shows what was attempted.

---

# NEXT-SESSION-PROMPT — `wip/apu-expert-pool`

Hand this verbatim to a cold session.  It assumes no memory of the previous one.

---

Continue the `wip/apu-expert-pool` campaign on this box (`halo` — Ryzen AI MAX+ 395, **gfx1151** UMA APU,
123 GiB, kernel 7.2.13, ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`).

**Read first, in this order:** `~/llama-cpp-rdna-boosts/AGENTS.md` (policy), then
`~/llama-cpp-rdna-boosts/wip/apu-expert-pool/HANDOVER.md` (**the cold-start document — every landmine and every
decision is in it**), then `README.md` §1 (the five gates), §10-§12 (dev logs) in the same directory.

**State:** the campaign has a working tier for a model that FITS — `-ncmoe 8 -lme auto -lm mmap` on
Qwen3.6-35B-A3B Q4_K_M gives byte-identical greedy text to `-ncmoe 0` with no GPU faults, `ROCm_HostMap` masters,
a pool-backed master, and a RAM-budgeted arena (75 % of one `MemAvailable` snapshot; the pool 10 %).  Four patches
(`0001`-`0004`) are pushed on the delivery branch `apu-expert-pool` and the fork branch `wip/apu-expert-pool`.
**The case the campaign exists for — a model LARGER than RAM — has never been run.**  That is your first job.

**Your first task (S1 in the handover, do this before anything else):** run the >RAM arm on
`/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/` (2 shards, 74 GiB, plus the shared `mtp-…-Q8_0.gguf`), at
`-ncmoe 24` and `-ncmoe 48`, with `-lme auto -lm mmap -c 32768`.  Record: greedy text hashed with **no `-lv`**
against the best fitting arm; `check-gpu-faults.sh` clean; the pool's **10 % ceiling binding** (the first config
where the host set can exceed it); `SLAB_INIT` ≤ the budget; `MemAvailable` after load above the OS reserve;
census with non-zero evictions and `takeover=0`; and confirm the "this model FITS, drop `-ncmoe`" advisory does
**not** fire.  If it faults, attribute it by address-to-buffer, never by kernel name.

Then work the handover's S2→S5 in order (unlock the read-only pool by skipping the UMA arena→pool demote; the
`scripts/gate-uma-arena.sh` promotion gate; the `--fit` reserve fix; the docs/housekeeping), reporting findings
to the maintainer **before** any commit that changes a default, per `AGENTS.md`.

**Ground rules that are easy to get wrong (all are in the handover's §7, abbreviated here):**

* `-lm mmap` is **mandatory** for the pool tier (an APU advertises no mmap support, so `auto` resolves to
  `none`); the maintainer has decided `caps.mmap_support` stays false and an APU declares its load mode.
* Build **fully** (`cmake --build build-wip -j 16`) and confirm with `nm -D` — a target-only build can leave
  `libggml-hip.so.0` with a stale undefined symbol and the binary then dies at startup.  Use `build-wip`; do not
  touch `build-rocm`.
* A new symbol in `moe-expert-cache.cu` must live **outside** its anonymous namespace (closes at `} // namespace`)
  or it has internal linkage and is dropped.
* Never test a budget for "explicitness" with `!g_auto` — `--fit` pre-fills `g_budget` from free VRAM.  Key on
  `MOE_EXPERT_CACHE_MIB`.
* Text arms: **no `-lv`**.  Censuses: `-lv 4`.  `-lv 3` is the trap.
* Two log lines in the UMA preflight arm never print (cosmetic, unexplained) — the census is authoritative; do
  not gate on those lines.
* Never push to upstream.  The delivery repo (`github.com:stew675/llama-cpp-rdna-boosts`) and the personal fork
  (`git@github.com:stew675/llama.cpp`) are the only push targets; the fork branch is `wip/apu-expert-pool`, never
  `rdna-boosts`.

Report findings and open questions to the maintainer before committing, not after.

---
