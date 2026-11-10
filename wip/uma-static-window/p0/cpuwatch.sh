#!/bin/bash
# Run one arm and report the process's mean CPU% over its decode window, plus the GPU's busy%.
# The question it answers: in an arm whose census says "99.5 % served cold by the GPU", is the CPU
# still computing MoE?  A GPU-only arm sits near 100 % CPU (one host thread submitting work); a
# CPU-computing arm sits at several hundred.
set -u
ARM=${1:?arm}; shift
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib HIP_VISIBLE_DEVICES=0
pid=""
samples=$(mktemp)
( while :; do
    if [ -z "$pid" ]; then pid=$(pgrep -n -f 'bin/llama-cli'); fi
    if [ -n "$pid" ] && [ -r "/proc/$pid/stat" ]; then
      awk '{print $14+$15}' "/proc/$pid/stat" >> "$samples"
      date +%s%N >> "$samples.clock"
      rocm-smi --showuse --json 2>/dev/null | grep -oE '"GPU use \(%\)": *"[0-9]+"' | grep -oE '[0-9]+' | head -1 >> "$samples.gpu"
    fi
    sleep 0.5
  done ) & W=$!
T0=$(date +%s%N)
./arms.sh "$ARM" "${N:-96}" > /tmp/cpuwatch-$ARM.out 2>&1
T1=$(date +%s%N)
kill $W 2>/dev/null
python3 - "$samples" "$T0" "$T1" "$ARM" <<'PY'
import sys
f,t0,t1,arm=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
try:
    j=[int(x) for x in open(f).read().split()]
    c=[int(x) for x in open(f+".clock").read().split()]
    g=[int(x) for x in open(f+".gpu").read().split()]
except Exception:
    print(arm,"sampler: n/a"); raise SystemExit
if len(j)>1 and len(c)>1:
    dt=(c[-1]-c[0])/1e9
    dc=(j[-1]-j[0])/100.0
    print(f"{arm:20s} mean CPU over the window: {dc/dt*100:6.1f} % of one core   (over {dt:.1f} s)")
if g: print(f"{arm:20s} mean GPU busy:            {sum(g)/len(g):6.1f} %")
PY
grep -oE "Generation: [0-9.]+ t/s" /tmp/cpuwatch-$ARM.out | tail -1
