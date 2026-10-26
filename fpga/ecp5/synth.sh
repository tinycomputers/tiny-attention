#!/usr/bin/env bash
# Phase 10: synthesize fpga_top for the ULX3S (Lattice ECP5 LFE5U-85F, CABGA381) and report utilization and fmax.
#   bash .tools/fpga.sh && bash fpga/ecp5/synth.sh
# Outputs go to fpga/ecp5/build/ (git-ignored): the netlist, the synthesis statistics, and the nextpnr report.
# Pin constraints are not included (no board here), so IO placement is left to nextpnr and fmax is an estimate.
set -euo pipefail
cd "$(dirname "$0")/../.."
Y=.tools/fpga-venv/bin/yowasp-yosys
N=.tools/fpga-venv/bin/yowasp-nextpnr-ecp5
B=fpga/ecp5/build
FREQ=${FREQ:-25}
mkdir -p "$B"

SRCS="rtl/lut_pkg.sv rtl/tiny_attn_pkg.sv $(ls rtl/*.sv | grep -v -e lut_pkg.sv -e tiny_attn_pkg.sv | tr '\n' ' ') fpga/ecp5/uart.sv fpga/ecp5/fpga_top.sv"

# 1. synthesis. The latch check runs on the elaborated design, before technology mapping.
"$Y" -q -l "$B/yosys.log" -p "read_verilog -sv $SRCS; hierarchy -check -top fpga_top; proc; \
  select -assert-none t:\$dlatch t:\$dlatchsr t:\$_DLATCH_* ; \
  check -assert; \
  synth_ecp5 -top fpga_top -json $B/fpga_top.json; \
  tee -o $B/stat.txt stat"
echo "synthesis: ok (no latches, no unresolved checks)"

# 2. place and route, with timing, against the ECP5 chip database
"$N" --85k --package CABGA381 --speed 6 --json "$B/fpga_top.json" --freq "$FREQ" \
  --report "$B/report.json" --timing-allow-fail -q > "$B/nextpnr.log" 2>&1 || true

python3 - "$B" <<'PY'
import json, re, sys
b = sys.argv[1]
stat = open(f"{b}/stat.txt").read()
def count(name):
    m = re.search(rf"^\s+(\d+)\s+{re.escape(name)}\b", stat, re.M)
    return int(m.group(1)) if m else 0
rep = json.load(open(f"{b}/report.json"))
fmax = None
for clk, v in rep.get("fmax", {}).items():
    fmax = (clk, v["achieved"], v["constraint"])
util = rep.get("utilization", {})
print("cells (synth):  LUT4", count("LUT4"), " TRELLIS_FF", count("TRELLIS_FF"),
      " MULT18X18D", count("MULT18X18D"), " TRELLIS_DPR16X4", count("TRELLIS_DPR16X4"),
      " DP16KD", count("DP16KD"), " PFUMX", count("PFUMX"), " CCU2C", count("CCU2C"))
for k in ("TRELLIS_COMB", "TRELLIS_FF", "MULT18X18D", "DP16KD", "DCUA"):
    if k in util:
        print(f"nextpnr {k}: used {util[k]['used']} of {util[k]['available']}")
if fmax:
    print(f"fmax: {fmax[1]:.1f} MHz achieved, constraint {fmax[2]:.1f} MHz (clock {fmax[0]})")
else:
    print("fmax: not reported (see nextpnr.log)")
PY
