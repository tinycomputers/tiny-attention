#!/usr/bin/env bash
# Run every cocotb unit suite, one SIM_BUILD each, deleted after its run (the build dirs are large and the
# disk is small). Usage: bash sim/run_unit.sh   (after `source scripts/env.sh`). End-to-end: sim/run_e2e.sh.
set -euo pipefail
cd "$(dirname "$0")"
# The installed Verilator leaves CFG_CXXFLAGS_PCH_I empty, so some generated models fail to link their precompiled header.
# -include is the GCC flag that rule needs (see sim/run_fpga.sh).
export MAKEFLAGS="CFG_CXXFLAGS_PCH_I=-include"
R=$PWD/../rtl
ARITH="$R/lut_pkg.sv $R/tiny_attn_pkg.sv $R/pe.sv $R/systolic_array.sv $R/mm_engine.sv $R/row_max.sv $R/scale_mask.sv $R/exp2_unit.sv $R/recip_unit.sv $R/online_softmax.sv $R/o_accumulator.sv $R/normalize.sv"
TILE="$ARITH $R/attn_tile.sv"
MEM="$R/sram_bank.sv $R/hbm_model.sv $R/dma.sv"
MM2="-GA_W=9 -GA_SIGNED=0 -GACC_W=18"

run() {  # name toplevel module sources [-G...]   (env A_W/A_SIGNED/ACC_W inherited)
  local name=$1 top=$2 mod=$3 srcs=$4; shift 4
  echo "=== $name"
  rm -f results.xml
  make MODULE="$mod" TOPLEVEL="$top" VERILOG_SOURCES="$srcs" WAVES=0 SIM_BUILD="sim_build/$name" EXTRA_ARGS="$*" >"sim_build_$name.log" 2>&1 || true
  python3 - "$name" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse("results.xml").getroot()
bad = [c.get("name") for c in r.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
n = len(list(r.iter("testcase")))
print(f"{sys.argv[1]}: {n - len(bad)}/{n} passed" + (f"  FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
PY
  rm -rf "sim_build/$name" "sim_build_$name.log"
}

run pe_mm1          pe             test_pe            "$ARITH"
A_W=9 A_SIGNED=0 ACC_W=18 run pe_mm2 pe test_pe "$ARITH" $MM2
run array_mm1       systolic_array test_systolic_array "$ARITH"
A_W=9 A_SIGNED=0 ACC_W=18 run array_mm2 systolic_array test_systolic_array "$ARITH" $MM2
run mm_engine_mm1   mm_engine      test_mm_engine     "$ARITH"
A_W=9 A_SIGNED=0 ACC_W=18 run mm_engine_mm2 mm_engine test_mm_engine "$ARITH" $MM2
run row_max         row_max        test_row_max       "$ARITH"
run scale_mask      scale_mask     test_scale_mask    "$ARITH"
run exp2            exp2_unit      test_exp2          "$ARITH"
run recip           recip_unit     test_recip         "$ARITH"
run online_softmax  online_softmax test_online_softmax "$ARITH"
run o_accumulator   o_accumulator  test_o_accumulator "$ARITH"
run normalize       normalize      test_normalize     "$ARITH"
run attn_tile       attn_tile      test_attn_tile     "$TILE"
run dma             dma_tb         test_dma           "$MEM $PWD/harness/dma_tb.sv"
echo "ALL UNIT SUITES PASSED"
