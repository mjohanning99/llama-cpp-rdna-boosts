#!/usr/bin/env python3
"""Achievable hit rate for a heat-ranked resident window, computed from the persisted MHT2 prior.

Format (ggml/src/ggml-cuda/moe-expert-cache.cu): header {magic[4], version u32, n_tables u32,
reserved0 u32, model_bytes u64, n_tensors u32, reserved1 u32} = 32 B; then per table
{layer i32, role char[64], n_experts u32, host_bytes u64} = 80 B followed by n_experts u32 of the
`all` band and n_experts u32 of the `dec` band.  `host_bytes` is the PER-EXPERT channel size.

Ranks every expert of every table by its decode-band count and reports, for each resident budget,
how much of the routed mass that budget covers -- i.e. the hit rate a perfect heat-ranked window gets.
"""
import struct, sys, array, collections

def load(path):
    b = open(path, 'rb').read()
    magic, ver, n_tab, r0, mbytes, n_ten, r1 = struct.unpack_from('<4sIIIQII', b, 0)
    assert magic in (b'MHT2', b'MHT1'), magic
    off = 32
    tabs = []
    for _ in range(n_tab):
        layer, role, n_exp, host_b = struct.unpack_from('<i64sIQ', b, off)
        off += 80
        allb = array.array('I'); allb.frombytes(b[off:off+4*n_exp]); off += 4*n_exp
        dec = None
        if ver >= 2:
            dec = array.array('I'); dec.frombytes(b[off:off+4*n_exp]); off += 4*n_exp
        tabs.append((layer, role.split(b'\0')[0].decode(), n_exp, host_b, allb, dec))
    return magic.decode(), ver, mbytes, tabs

path = sys.argv[1]
magic, ver, mbytes, tabs = load(path)
te = sum(t[2] for t in tabs)
tb = sum(t[2]*t[3] for t in tabs)
print(f"{path}\n  format {magic} v{ver}, {len(tabs)} tables, {te} experts, "
      f"{tb/2**30:.1f} GiB of expert bytes, per-expert {min(t[3] for t in tabs)/2**20:.2f}-"
      f"{max(t[3] for t in tabs)/2**20:.2f} MiB")
# global ranking by the decode band (fall back to `all` for v1)
recs = []
use_dec = all(t[5] is not None for t in tabs)
for (layer, role, n_exp, host_b, allb, dec) in tabs:
    band = dec if (use_dec and dec is not None) else allb
    for e in range(n_exp):
        recs.append((band[e], host_b))
recs.sort(key=lambda x: -x[0])
total = sum(r[0] for r in recs)
print(f"  ranked by the {'decode' if use_dec else 'all'} band; total routed mass = {total}")
if total == 0:
    print("  EMPTY prior -- nothing to rank"); sys.exit()
# also per-table byte share and how uniform the mass is across tables
per_table = [sum((t[5] if (use_dec and t[5]) else t[4])) for t in tabs]
print(f"  mass per table: min {min(per_table)} max {max(per_table)} ratio {max(per_table)/max(1,min(per_table)):.2f}"
      "  (a ratio near 1 means the mass is uniform per table -- i.e. LAYER placement cannot concentrate it)")
print()
print(f"{'resident GiB':>12} {'% of bytes':>10} {'hit rate':>9} {'miss/token GiB':>15}")
byts = 0; mass = 0
targets = [int(x) for x in (sys.argv[2:] or ['8','24','48','64','80','96','104','112'])]
ti = 0
for c, hb in recs:
    byts += hb; mass += c
    gib = byts/2**30
    while ti < len(targets) and gib >= targets[ti]:
        miss = (total-mass)/total*3.8 if total else 0
        print(f"{targets[ti]:12d} {byts/tb*100:9.1f}% {mass/total*100:8.2f}% {miss:15.3f}")
        ti += 1
