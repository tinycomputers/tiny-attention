#!/usr/bin/env bash
# Phase 10 simulation: the FPGA top driven over its UART (fpga/ecp5/fpga_top.sv), at 4 clocks per bit to keep it quick.
# Usage: bash sim/run_fpga.sh   (after `source scripts/env.sh`)
set -euo pipefail
cd "$(dirname "$0")"
R=$PWD/../rtl
F=$PWD/../fpga/ecp5
SRCS="$R/lut_pkg.sv $R/tiny_attn_pkg.sv $(ls $R/*.sv | grep -v -e lut_pkg.sv -e tiny_attn_pkg.sv | tr '\n' ' ') $F/uart.sv $F/fpga_top.sv"
rm -f results.xml
# The installed Verilator leaves CFG_CXXFLAGS_PCH_I empty, so its precompiled-header rule passes a file name to the
# compiler. -include is the GCC flag the rule needs.
MAKEFLAGS="CFG_CXXFLAGS_PCH_I=-include" make MODULE=test_fpga_uart TOPLEVEL=fpga_top VERILOG_SOURCES="$SRCS" \
  WAVES=0 SIM_BUILD=sim_build/fpga EXTRA_ARGS="-GCLKS_PER_BIT=4" > sim_build_fpga.log 2>&1 || true
python3 - <<'PY'
import xml.etree.ElementTree as ET
r = ET.parse("results.xml").getroot()
bad = [c.get("name") for c in r.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
n = len(list(r.iter("testcase")))
print(f"fpga uart: {n - len(bad)}/{n} passed" + (f"  FAILED: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
PY
rm -rf sim_build/fpga sim_build_fpga.log
