#!/usr/bin/env bash
# End-to-end engine test: tiny_attn_top vs the golden model, byte counts, tile counts and cycle model.
# SEEDS=200 by default (about 200 x 16 runs). Usage: bash sim/run_e2e.sh   (after `source scripts/env.sh`)
set -euo pipefail
cd "$(dirname "$0")"
R=$PWD/../rtl
SRCS="$R/lut_pkg.sv $R/tiny_attn_pkg.sv $(ls $R/*.sv | grep -v -e lut_pkg.sv -e tiny_attn_pkg.sv | tr '\n' ' ')"
rm -f results.xml
make MODULE=test_e2e TOPLEVEL=tiny_attn_top VERILOG_SOURCES="$SRCS" WAVES=0 SIM_BUILD=sim_build/e2e \
  >sim_build_e2e.log 2>&1 || true
python3 - <<'PY'
import xml.etree.ElementTree as ET
r = ET.parse("results.xml").getroot()
bad = [c.get("name") for c in r.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
n = len(list(r.iter("testcase")))
print(f"e2e: {n - len(bad)}/{n} passed" + (f"  FAILED: {bad}" if bad else ""))
PY
rm -rf sim_build/e2e sim_build_e2e.log
