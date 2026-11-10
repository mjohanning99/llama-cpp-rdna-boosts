# NEXT-SESSION-PROMPT — `wip/uma-static-window`

Hand this verbatim to a cold session.  It assumes no memory of any previous session.

---

You are starting the `wip/uma-static-window` campaign on this box (`halo` — Ryzen AI MAX+ 395, **gfx1151** UMA
APU, 123 GiB, kernel 7.2.13, ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`).  It is a **clean start with a defined
direction**, not a continuation: the campaign this replaces was closed on 2026-11-10.

**Read first, in this order:**

1. `~/llama-cpp-rdna-boosts/AGENTS.md` (policy — apply, push, gates, default-on, reporting rules).
2. `~/llama-cpp-rdna-boosts/wip/uma-static-window/README.md` (**the design, the arithmetic, the plan, the
   landmines**).
3. `~/llama-cpp-rdna-boosts/archive/work/apu-expert-pool/CONCLUSION.md` (**the negative results that produced this
   design** — read it so you do not re-walk the arena path; its `README.md`/`IMPLEMENTATION.md` describe a design
   that was measured to be 3x slower and are kept only as the record).

**The design in one line:** on a shared host/device pool, select the hottest experts **once** by the persisted heat
prior, copy them **once** into an anonymous buffer, **then** register that buffer read-only for the GPU, publish the
expert→slot remap **once**, and let the GPU read the window **in place** while the complement stays on the CPU expert
path.  No arena, no eviction, no demote, no per-op host round-trip.  `-lme N` (GiB) is the window size; `-ncmoe` is a
no-op on UMA with one WARN; a model that fits is left alone (advisory only); a model that does not fit **declares
heat-map mode loudly**.

**Your first task is P0, and it is a decision, not code.**  Answer, with measurements:

- **P0.1** — on `Qwen3.6-35B-A3B` (`/llm/models/Qwen3.6/35B-A3B/Q4_K_M/`), can *every* expert be served in place
  from a registered pool (no arena residency, `MOE_HOST_POOL_COLD`) at least at CPU-path parity?
- **P0.2** — can one routed op mix *some* experts in a GPU-readable window with the rest on the CPU, and what does
  the split cost per op?  (The id encoding `n_res_slots + e` already exists; the fusion behaviour for a mixed table
  is the open question.  If ggml cannot express it, that alone re-scopes the campaign.)
- **P0.3** — can the remap be uploaded once at load and left alone (no per-token readback) through a captured
  decode graph?

**Stop rule:** if P0.1 or P0.2 shows the in-place/mixed path is not faster than the CPU path at full residency, stop
and write the negative result into this campaign's `README.md` and a `WORKLOG.md` entry.  That is a successful
outcome.

**Numbers to beat** (measured 2026-11-09/10 on this box, r243 + the inherited fixes):

| case | reference |
|---|---|
| Qwen3.5-122B-A10B Q8_0 (`-ncmoe 48 -c 32768`), CPU expert path | **10.58 / 10.90 / 10.89 t/s** — the number the design must beat |
| the same model, the old arena tier (32 / 70 GiB) | 3.75 / 3.82 / 3.04 and 3.29 / 2.84 / 2.36 — **do not reproduce this** |
| Qwen3.8-Flash-Next IQ4_NL 87 GiB + MTP drafter, 31 GiB ballast, `-ncmoe 24`, old arena 21 GiB | 37.4 / 43.6 / **44.0 t/s** — the control that must not regress |
| the same, CPU path | 29.7 / 30.8 / 31.0 t/s |
| GPU read of a registered-anon window / of the arena | **198.7** / 194.3 GB/s (probe `archive/work/apu-expert-pool/probe/11-fill-path.cpp`) |

**Ground rules that are easy to get wrong:**

* **Fill, then register.**  Register-then-fill runs at 70–210 MB/s (page-fault bound); ~7 GB/s the other way.
* **Check that the tier actually armed.**  Below the slab's work+floor+ring minimum (~9 GiB for a 122B) it prints
  `STREAMING the experts from the host` and the run is the CPU path; three arms were misread as tier arms this way.
  Look for `table_pool_locked` / the census, never the t/s alone.
* **The exit census does not reliably print** on `llama-server` `SIGTERM` (1 of 6 arms).  Use `llama-cli` for a
  census or add a mid-run report.
* Text arms: **no `-lv`**.  Censuses: `-lv 4`.  `-lv 3` is the trap.  `HIP_VISIBLE_DEVICES=0` on every arm.
* Build **fully** (`cmake --build build-wip -j 16`) and confirm with `nm -D`; a target-only build can leave a stale
  undefined symbol.  Use `build-wip`; do not touch `build-rocm`.
* **Never leave a workload unsupervised.**  Run the server in its own session, watch the process (not just the URL),
  give every request a `timeout`, and kill the whole group.  Two runs in the previous campaign hung for 10–24
  minutes on a dead server; one orphaned process took the desktop down with it.
* Never push to upstream.  The delivery repo (`github.com:stew675/llama-cpp-rdna-boosts`) and the personal fork
  (`git@github.com:stew675/llama.cpp`, branch `wip/uma-static-window`) are the only push targets.

**Report findings and open questions to the maintainer before committing anything that changes a default**, per
`AGENTS.md` — and before you commit at all, if a finding changes the design.
