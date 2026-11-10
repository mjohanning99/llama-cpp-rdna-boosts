#!/bin/bash
# 122B arms for wip/uma-static-window P0/P1: where should the expert bytes LIVE on a UMA APU?
#
# On this box `hipMalloc` is GTT -- the same DRAM as host RAM -- and p0/band.cpp measures a
# registered-host window at 231 GB/s against hipMalloc's 232 GB/s (identical at every launch shape).
# So "pin a window of the hottest experts" and "put those experts on the device" are the SAME memory at
# the SAME speed, and the second one needs no copy.  These arms measure the thing that actually varies:
# how much of the expert set is on the DEVICE (-ncmoe N keeps the last N layers' experts on the host,
# so a SMALL N = most experts on the device).
#
#   ./arms122.sh <n_cpu_layers> [n_predict]
#   DEV=1 ./arms122.sh 24          # the same, but every layer's experts on the device (no host part)
#
# A memory sampler runs alongside (MemAvailable / SwapUsed, max), because the failure mode here is not
# a slow arm -- it is a swap storm, and the run must be killed before it takes the desktop down.
set -u
NCMOE=${1:?n_cpu_layers}
NP=${2:-64}
MODEL=/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
CLI=/home/stew675/llama.cpp/build-wip/bin/llama-cli
EXTRACT=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
OUT=/home/stew675/llama-cpp-rdna-boosts/wip/uma-static-window/p0/results
STAMP=$(date +%y%m%d-%H%M%S)
DIR="$OUT/${STAMP}-122b-ncmoe${NCMOE}"; mkdir -p "$DIR"
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0

# The host-expert placement: `pool` (pageable master + page cache, CPU computes the host layers) is
# the only mode that does not duplicate the bytes on an APU -- 114.8 GiB of experts cannot be pinned
# twice on a 123 GiB box.  The cache is OFF: these arms are about placement, not about the tier.
ENVV=(MOE_EXPERT_CACHE_MIB=0)
if [ "${DEV:-0}" = 1 ]; then
  ARGS=(-ncmoe 0)
else
  ARGS=(-ncmoe "$NCMOE" --host-experts pool -lm mmap)
fi

COMMON=(-m "$MODEL" -ngl 99 -c 8192 -p "The capital of France is" -n "$NP"
        --seed 42 --temp 0 --single-turn --no-display-prompt -t 12)

sampler() { # $1 = file
  : > "$1"
  while :; do
    awk '/MemAvailable|SwapTotal|SwapFree/ {printf "%s=%s ", $1, $2}' /proc/meminfo >> "$1"
    printf "%s\n" "$(date +%H:%M:%S)" >> "$1"
    sleep 2
  done
}
sampler "$DIR/mem.log" & SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

run() {
  local tag=$1; shift
  echo "=== 122B ncmoe=$NCMOE (${DEV:-0}) / $tag ==="
  ( set -x; env -- "${ENVV[@]}" "$CLI" "${COMMON[@]}" "${ARGS[@]}" "$@" </dev/null ) \
    > "$DIR/$tag.log" 2>&1
  echo "rc=$?  log=$DIR/$tag.log"
  grep -E "^\[ Prompt:" "$DIR/$tag.log" | tail -1
  [ "$tag" = text ] && { python3 "$EXTRACT" "$DIR/$tag.log" --text > "$DIR/$tag.txt" 2>/dev/null &&
    printf 'hash: '; python3 "$EXTRACT" "$DIR/$tag.log" 2>/dev/null; }
}

run text
run census -lv 4
kill $SPID 2>/dev/null; trap - EXIT

echo "--- memory (min MemAvailable MiB, max swap used MiB) ---"
awk '{for(i=1;i<=NF;i++){if($i ~ /^MemAvailable:/){v=$(i+1)}; if($i ~ /^SwapTotal:/){t=$(i+1)}; if($i ~ /^SwapFree:/){f=$(i+1)}};
      if(v!="" && (mn==""||v<mn)) mn=v; if(t!=""&&f!="" && (t-f)>mx) mx=t-f}
     END{printf "min MemAvailable=%.0f MiB   max swap used=%.0f MiB\n", mn, mx}' "$DIR/mem.log"
echo "--- 122B keys ---"
grep -iE "STREAMING|tables=|host experts:|FITS|summar|warning|error" "$DIR/census.log" \
  | grep -viE "Werror|no-display" | head -12
echo "dir=$DIR"
