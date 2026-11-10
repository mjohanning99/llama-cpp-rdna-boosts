#!/bin/bash
# P1 -- the capacity-constrained rig: does a STATIC READ-ONLY window beat the shipped copy tier at equal
# pinned bytes?  31 GiB of ballast (wip/bench-56gb/ballast, mlock'd outside any cgroup) puts MemAvailable at
# ~85 GiB, so a 96 GiB IQ4_NL model cannot keep its host expert set in the page cache -- which is the only
# configuration on this box where a pinned tier has slack to work with (the closed campaign measured the
# copy tier at 44.0 t/s there against the CPU path's 31.0).
#
#   ./p1.sh <arm> [n_predict]
#     cpu-path    -ncmoe 24, host experts in the page cache, cache OFF   (the CPU reference)
#     tier47      -lme 47, the SHIPPED tier: a covering pool + arena, default admission, copy per miss
#     window47    -lme 47, the STATIC WINDOW: covering pool registered READ-ONLY, one arena slot per table,
#                 admission never admits -> every routed expert read in place, no fills, no evictions
#     window21    the same at 21 GiB (match the recorded 44.0 t/s arm's bytes)
#     tier21      the shipped tier at 21 GiB (the 44.0 reference itself)
#
# The ballast runs in its own session and is killed by process group, always (a leaked mlock is a wedged box).
set -u
ARM=${1:?arm}
N=${2:-96}
P0=/home/stew675/llama-cpp-rdna-boosts/wip/uma-static-window/p0
MODEL=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
CLI=/home/stew675/llama.cpp/build-wip/bin/llama-cli
EXTRACT=/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py
OUT=$P0/results
STAMP=$(date +%y%m%d-%H%M%S)
DIR="$OUT/${STAMP}-p1-${ARM}"; mkdir -p "$DIR"
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0

# a covering pool that is READ-ONLY is only legal with the demote off; the arena must exist for a host table
# to register at all (a zero-arena cache registers nothing), so the window arm keeps ONE slot per table.
COVER=(MOE_HOST_POOL_FRAC=100 MOE_HOST_POOL_RAM_PCT=0 MOE_HOST_POOL_SINK=0)
case "$ARM" in
  cpu-path) ENVV=(MOE_EXPERT_CACHE_MIB=0);                                LME=24 ;;
  tier47)   ENVV=(MOE_HOST_POOL_SINK=1 MOE_HOST_POOL_RAM_PCT=0);          LME=47 ;;
  tier21)   ENVV=(MOE_HOST_POOL_SINK=1 MOE_HOST_POOL_RAM_PCT=0);          LME=21 ;;
  window47) ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB=32768
                  GGML_CUDA_SLAB_RESERVE_MIB=1024 MOE_EXPERT_CACHE_SLOTS=1 MOE_EXPERT_CACHE_TOUCH=1000000)
            LME=47 ;;
  window21) ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB=32768
                  GGML_CUDA_SLAB_RESERVE_MIB=1024 MOE_EXPERT_CACHE_SLOTS=1 MOE_EXPERT_CACHE_TOUCH=1000000)
            LME=21 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

# ballast: own session, so one kill ends it
mem_before=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
setsid /home/stew675/llama-cpp-rdna-boosts/wip/bench-56gb/ballast 31 > "$DIR/ballast.log" 2>&1 &
BPID=$!
for _ in $(seq 1 60); do grep -q locked "$DIR/ballast.log" 2>/dev/null && break; sleep 0.25; done
mem_after=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "ballast pid=$BPID  MemAvailable ${mem_before} -> ${mem_after} MiB"
trap 'kill -- -$BPID 2>/dev/null; kill $BPID 2>/dev/null' EXIT

COMMON=(-m "$MODEL" -ngl 99 -c 8192 -p "The capital of France is" -n "$N"
        --seed 42 --temp 0 --single-turn --no-display-prompt -t 12 -ncmoe 24 --host-experts pool -lm mmap)
[ "$ARM" != cpu-path ] && COMMON+=(-lme "$LME")

run() {
  local tag=$1; shift
  echo "=== p1 $ARM / $tag (ballast up) ==="
  ( set -x; env -- "${ENVV[@]}" "$CLI" "${COMMON[@]}" "$@" </dev/null ) > "$DIR/$tag.log" 2>&1
  echo "rc=$?  log=$DIR/$tag.log"
  grep -E "^\[ Prompt:" "$DIR/$tag.log" | tail -1
  [ "$tag" = text ] && { printf 'hash: '; python3 "$EXTRACT" "$DIR/$tag.log" 2>/dev/null; }
  awk '/MemAvailable/{v=int($2/1024)} /SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "  after %s: MemAvailable=%d MiB swap_used=%d MiB\n", "'"$tag"'", v, (t-f)/1024}' /proc/meminfo
}
if [ "${ONLY:-both}" = text ]; then run text; else run text; run census -lv 4; fi
kill -- -$BPID 2>/dev/null; kill $BPID 2>/dev/null; trap - EXIT
sleep 1
echo "ballast released; MemAvailable=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
echo "--- tier keys ---"
grep -iE "host experts:|STREAMING|tables=|expert access:|pool \(L2\)|summar|FITS|STREAM" "$DIR/census.log" \
  | grep -viE "Werror|no-display" | head -10
echo "dir=$DIR"
