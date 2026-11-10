#!/bin/bash
# 122B WINDOW arms for wip/uma-static-window P2 -- the arm the campaign's arithmetic predicts and that has
# never been run: a heat-ranked, zero-copy window of the model's own expert pages, sized to a resident budget,
# measured against the CPU expert path's 8.0 t/s (marginal, tokens 64..320) on this box.
#
#   ./window122.sh <GiB> [n_predict] [mode]
#     ./window122.sh 88 64            # decode: an 88 GiB window, -n 64
#     ./window122.sh 33 1 prefill     # prefill: a 33 GiB window, 4059-token prompt, -b/-ub 4096
#
# Why a script and not a bare command: every one of these arms pins tens of GiB, and the failure mode is not a
# slow arm -- it is a swap storm that takes the desktop down.  So the run is bounded (`timeout -k`), it is its own
# process group (`setsid` so one kill ends it), and a sampler records MemAvailable / SwapUsed every 2 s.  The
# t/s is read from llama's own `[ Prompt: ... | Generation: ... ]` footer; the TEXT arm carries no `-lv` at all.
set -u
GIB=${1:?window GiB}
NP=${2:-64}
MODE=${3:-decode}
MODEL=${MODEL:-/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf}
CLI=${CLI:-/home/stew675/llama.cpp/build-wip/bin/llama-cli}
EXTRACT=${EXTRACT:-/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py}
PROMPT122=${PROMPT122:-/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt}
OUT=${OUT:-/home/stew675/llama-cpp-rdna-boosts/wip/uma-static-window/p0/results}
STAMP=$(date +%y%m%d-%H%M%S)
DIR="$OUT/${STAMP}-122b-window${GIB}g-${MODE}"
mkdir -p "$DIR"
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib
export HIP_VISIBLE_DEVICES=0

MIB=$(( GIB * 1024 ))

# The window is the pool's memory type, so it is sized with the pool's own budget knob (`MOE_HOST_POOL_MIB`,
# split across the host tensors in proportion to their expert bytes, i.e. one heat-ranked window per table).
# `MOE_HOST_POOL_RAM_PCT=0` lifts the UMA ceiling: this branch defaults it to 10 % of MemAvailable (11.97 GiB),
# which would silently scale a 96 GiB plan by 0.122 -- any "the tier cannot hold N" result must first confirm the
# clamp did not fire (RESULTS §H).  SINK=0 because a window has no arena->pool demote (it is read-only).
ENVV=(MOE_HOST_POOL_ALLOC=window MOE_HOST_POOL_MIB="$MIB" MOE_HOST_POOL_RAM_PCT=0 MOE_HOST_POOL_SINK=0
      MOE_HOST_POOL_FRAC=100
      MOE_EXPERT_CACHE_MIB=24576 GGML_CUDA_SLAB_RESERVE_MIB=1024
      MOE_EXPERT_CACHE_SLOTS=1 MOE_EXPERT_CACHE_TOUCH=1000000)

if [ "$MODE" = prefill ]; then
  # The prefill baseline to beat (RESULTS §J): 4059 tokens, -b 4096 -ub 4096 -> 175.2 t/s (23.17 s), of which
  # 9.98 s is the pool's fill memcpy (33.15 GiB at 3772 MiB/s).
  COMMON=(-m "$MODEL" -ngl 99 -c 8192 -b 4096 -ub 4096 -f "$PROMPT122" -n "$NP"
          --seed 42 --temp 0 --single-turn --no-display-prompt -t 12 -ncmoe 48 --host-experts pool -lm mmap)
else
  COMMON=(-m "$MODEL" -ngl 99 -c 4096 -p "The capital of France is" -n "$NP"
          --seed 42 --temp 0 --single-turn --no-display-prompt -t 12 -ncmoe 48 --host-experts pool -lm mmap)
fi

sampler() {
  : > "$1"
  while :; do
    awk '/^MemAvailable|^SwapTotal|^SwapFree/ {printf "%s=%s ", $1, $2}' /proc/meminfo >> "$1"
    date +%H:%M:%S >> "$1"
    sleep 2
  done
}
sampler "$DIR/mem.log" & SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

echo "=== 122B / ${GIB} GiB window / $MODE / -n $NP ==="
setsid timeout -k 5 1800 env -- "${ENVV[@]}" "$CLI" "${COMMON[@]}" -lv 4 </dev/null > "$DIR/run.log" 2>&1
echo "rc=$?  log=$DIR/run.log"
grep -E "^\[ Prompt:" "$DIR/run.log" | tail -1
kill $SPID 2>/dev/null; trap - EXIT

echo "--- memory (min MemAvailable MiB, max swap used MiB) ---"
awk '{for(i=1;i<=NF;i++){if($i ~ /^MemAvailable:/){v=$(i+1)}; if($i ~ /^SwapTotal:/){t=$(i+1)}; if($i ~ /^SwapFree:/){f=$(i+1)}};
      if(v!="" && (mn==""||v<mn)) mn=v; if(t!=""&&f!="" && (t-f)>mx) mx=t-f}
     END{printf "min MemAvailable=%.0f MiB   max swap used=%.0f MiB\n", mn, mx}' "$DIR/mem.log"
echo "--- keys ---"
grep -E "WINDOW pool|host pool \(L2\)|expert access|prompt eval|device remap|STREAMING|tables=" "$DIR/run.log" | head -12
echo "dir=$DIR"
