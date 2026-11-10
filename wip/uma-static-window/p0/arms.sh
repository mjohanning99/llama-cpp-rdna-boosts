#!/bin/bash
# P0 arms for wip/uma-static-window -- "the facts before the code".
#
# One arm = one llama-cli load.  A TEXT arm carries no -lv (the stream is the generated text,
# hashed with scripts/extract-generated.py); a CENSUS arm repeats it at -lv 4 (a separate run,
# because -lv 3 both pollutes the text and hides the ggml censuses -- see AGENTS.md).
#
#   ./arms.sh <arm> [n_predict]
#
# Arms (35B-A3B Q4_K_M unless the model env overrides it):
#   ref-device   -ncmoe 0                                -- every expert on the device (the "fits" ceiling)
#   cpu-path     -ncmoe 40, pool placement, cache OFF    -- the CPU expert path (the number to beat)
#   cold100      every routed op served in place from the pool (arena armed, nothing ever admitted)
#   cold100-wr   the same with the pool's device alias registered WRITABLE (the RO A/B)
#   mixNN        arena sized to ~NN% of the expert set + a covering pool: a genuine MIXED routed op
#   pinned-set   -ncmoe 40, --host-experts pinned        -- refused on this box (the r243 default policy)
#
# Every arm pins HIP_VISIBLE_DEVICES (an iGPU is enumerated on this box and a bare split enumerates it).
set -u

ARM=${1:?arm}
N=${2:-96}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf}
CLI=${CLI:-/home/stew675/llama.cpp/build-wip/bin/llama-cli}
EXTRACT=${EXTRACT:-/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py}
OUT=${OUT:-/home/stew675/llama-cpp-rdna-boosts/wip/uma-static-window/p0/results}
STAMP=$(date +%y%m%d-%H%M%S)
DIR="$OUT/${STAMP}-${ARM}"
mkdir -p "$DIR"

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib
export HIP_VISIBLE_DEVICES=0

COMMON=(-m "$MODEL" -ngl 99 -c 4096 -p "The capital of France is" -n "$N"
        --seed 42 --temp 0 --single-turn --no-display-prompt -t 12)

# MOE_HOST_POOL_SINK=0 + a covering pool: the arena->pool demote (the only device-side writer of the
# pool backing) is off, which is what makes a READ-ONLY registration legal (moe-expert-cache.cu).
COVER=(MOE_HOST_POOL_FRAC=100 MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_POOL_SINK=0)

case "$ARM" in
  ref-device)
    ENVV=(); ARGS=(-ncmoe 0)
    ;;
  cpu-path)
    ENVV=(MOE_EXPERT_CACHE_MIB=0); ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  cold100)
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB="${MIB:-6144}"
          GGML_CUDA_SLAB_RESERVE_MIB="${RESERVE:-1024}" GGML_CUDA_SLAB_MIN_ARENA_MIB="${MIN_ARENA:-64}"
          MOE_EXPERT_CACHE_TOUCH=1000000)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  cold100-strict)
    # A SIZE-TUNED arm: the slab is big enough for every table to register (cap 24 GiB), but each table
    # gets ONE slot and the admission never admits (TOUCH=huge), so ~100% of routed experts are served in
    # place from the covering pool.  This is P0.1 done properly: all 120 tables GPU-served, no CPU part.
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB="${MIB:-24576}"
          GGML_CUDA_SLAB_RESERVE_MIB="${RESERVE:-1024}"
          MOE_EXPERT_CACHE_SLOTS="${SLOTS:-1}" MOE_EXPERT_CACHE_TOUCH=1000000)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  cold100-strict-wr)
    # The registration A/B for the arm above: same shape, the pool's device alias registered WRITABLE
    # (60 GB/s against 233 GB/s measured by the probe on this box).  If the two arms tie, decode is not
    # read-band limited here and the 35B cannot show the band win; if they separate, the band is the lever.
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=0 MOE_EXPERT_CACHE_MIB="${MIB:-24576}"
          GGML_CUDA_SLAB_RESERVE_MIB="${RESERVE:-1024}"
          MOE_EXPERT_CACHE_SLOTS="${SLOTS:-1}" MOE_EXPERT_CACHE_TOUCH=1000000)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  cold100-wr)
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=0 MOE_EXPERT_CACHE_MIB="${MIB:-6144}"
          GGML_CUDA_SLAB_RESERVE_MIB="${RESERVE:-1024}" GGML_CUDA_SLAB_MIN_ARENA_MIB="${MIN_ARENA:-64}"
          MOE_EXPERT_CACHE_TOUCH=1000000)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  window*)
    # windowNN: a HEAT-RANKED window of NN% of each table's experts in the arena, the complement served in
    # place from the covering pool -- all 120 tables GPU-served, ONE routed op per layer carrying both id
    # encodings (slot for a resident expert, n_res + pool slot for the rest).  Default admission (touch/2),
    # so the residents converge on the hot experts exactly as a static heat-prior window would pick them.
    PCT=${ARM#window}; PCT=${PCT%%[!0-9]*}
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB="${MIB:-24576}"
          GGML_CUDA_SLAB_RESERVE_MIB="${RESERVE:-1024}"
          MOE_EXPERT_CACHE_SLOTS=$(( 256 * PCT / 100 ))
          MOE_EXPERT_CACHE_ADMIT=2 MOE_EXPERT_CACHE_TOUCH=2)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  pinned-set)
    ENVV=(); ARGS=(-ncmoe 40 --host-experts pinned)
    ;;
  mix*)
    # mixNN [MIB]: an arena of MIB, a covering pool, the default admission (touch/2) -- so the
    # hottest experts become resident and the rest are served in place from the pool.  One routed
    # op then carries BOTH encodings (slots + e for the resident ones, n_res + pool slot for the rest).
    PCT=${ARM#mix}; PCT=${PCT%%[!0-9]*}
    MIB=${3:-$(( 20480 * PCT / 100 ))}
    ENVV=("${COVER[@]}" MOE_HOST_POOL_RO=1 MOE_EXPERT_CACHE_MIB="$MIB"
          MOE_EXPERT_CACHE_ADMIT=2 MOE_EXPERT_CACHE_TOUCH=2
          MOE_EXPERT_CACHE_MIN_RES_PCT=0 MOE_EXPERT_CACHE_IDENTITY_PREF=0)
    ARGS=(-ncmoe 40 --host-experts pool -lm mmap)
    ;;
  *)
    echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

run() { # $1 = tag, rest = extra llama-cli args
  local tag=$1; shift
  echo "=== $ARM / $tag ==="
  ( set -x
    env -- "${ENVV[@]}" "$CLI" "${COMMON[@]}" "${ARGS[@]}" "$@" </dev/null
  ) > "$DIR/$tag.log" 2>&1
  local rc=$?
  echo "rc=$rc  log=$DIR/$tag.log"
  grep -E "^\[ Prompt:|llama_perf_context_print: prompt eval|^llama_perf_context_print: *eval time" "$DIR/$tag.log" | tail -3
  if [ "$tag" = text ]; then
    python3 "$EXTRACT" "$DIR/$tag.log" --text > "$DIR/$tag.txt" 2>"$DIR/$tag.err" &&
    { echo -n "hash: "; python3 "$EXTRACT" "$DIR/$tag.log"; } || cat "$DIR/$tag.err"
  fi
}

# The text arm: no -lv at all.
run text
# The census arm: a separate run.
run census -lv 4

echo
echo "--- census keys ---"
grep -iE "table_pool_locked|STREAMING the experts|cold|arena|pool|takeover|get_ok|devremap|registered" \
     "$DIR/census.log" | grep -viE "^llama_model_loader: - kv|^llama_model_load|tokenizer" | head -40
echo
echo "dir=$DIR"
