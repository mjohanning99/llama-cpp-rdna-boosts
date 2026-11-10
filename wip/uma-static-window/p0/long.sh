#!/bin/bash
# Long-generation arms: does the hot set establishing itself show up as rising throughput?
# Runs the SAME arm twice (-n 96 and -n 512) and derives the MARGINAL rate over tokens (96, 512]:
#   marginal = (512-96) / (t512 - t96), with t = n / (the footer's Generation t/s).
# `EXTRA_ENV="A=1 B=0"` adds environment for this arm (e.g. MOE_EXPERT_HEATMAP=0 for a cold start).
set -u
ARM=${1:?arm}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0
export ${EXTRA_ENV:-}
P0=/home/stew675/llama-cpp-rdna-boosts/wip/uma-static-window/p0
OUT=$P0/results
STAMP=$(date +%y%m%d-%H%M%S)
TAG=${TAG:-$ARM}
DIR="$OUT/${STAMP}-long-${TAG}"; mkdir -p "$DIR"
for n in 96 512; do
  cp -a "$P0/arms.sh" /tmp/arms-long.sh   # arms.sh inherits EXTRA_ENV through the environment
  ( cd "$P0" && ./arms.sh "$ARM" "$n" ) > "$DIR/run-$n.out" 2>&1
  d=$(grep -oE 'log=[^ ]*' "$DIR/run-$n.out" | head -1 | sed 's|^log=||; s|/[^/]*$||')
  cp "$d/text.log" "$DIR/text-$n.log" 2>/dev/null
  cp "$d/census.log" "$DIR/census-$n.log" 2>/dev/null
  rm -rf "$d"
done
echo "--- $TAG ${EXTRA_ENV:-} ---"
for n in 96 512; do echo "n=$n  $(grep -E 'Generation:' "$DIR/text-$n.log" | tail -1)"; done
python3 - "$DIR" <<'PY'
import re,sys
d=sys.argv[1]
r={}
for n in (96,512):
    s=open(f"{d}/text-{n}.log",errors="replace").read()
    m=re.findall(r"Generation:\s*([0-9.]+) t/s", s)
    if not m: print("no footer for n=",n); sys.exit(1)
    r[n]=float(m[-1])
t1,t2=96/r[96],512/r[512]
print(f"whole-run {r[96]:.1f} t/s (n=96) -> {r[512]:.1f} t/s (n=512)")
print(f"marginal over tokens 96..512: {416/(t2-t1):.1f} t/s  ({'+' if 416/(t2-t1)>r[96] else ''}{100*(416/(t2-t1)/r[96]-1):.1f}% vs the first 96)")
PY
echo "dir=$DIR"
