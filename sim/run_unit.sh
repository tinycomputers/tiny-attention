#!/usr/bin/env bash
# Run every cocotb unit suite, one SIM_BUILD each (cocotb does not notice a TOPLEVEL/parameter change across
# runs that share a build dir). Usage: bash sim/run_unit.sh   (after `source scripts/env.sh`)
set -euo pipefail
cd "$(dirname "$0")"
SRC="$PWD/../rtl/lut_pkg.sv $PWD/../rtl/tiny_attn_pkg.sv $PWD/../rtl/pe.sv $PWD/../rtl/systolic_array.sv \
$PWD/../rtl/mm_engine.sv $PWD/../rtl/row_max.sv $PWD/../rtl/scale_mask.sv $PWD/../rtl/exp2_unit.sv $PWD/../rtl/recip_unit.sv \
$PWD/../rtl/online_softmax.sv $PWD/../rtl/o_accumulator.sv $PWD/../rtl/normalize.sv $PWD/../rtl/attn_tile.sv"
MM2="-GA_W=9 -GA_SIGNED=0 -GACC_W=18"

run() {  # name  toplevel  module  [compile args]  (env A_W/A_SIGNED/ACC_W inherited)
  local name=$1 top=$2 mod=$3; shift 3
  echo "=== $name"
  rm -f results.xml
  make MODULE="$mod" TOPLEVEL="$top" VERILOG_SOURCES="$SRC" WAVES=0 SIM_BUILD="sim_build/$name" EXTRA_ARGS="$*" 2>&1 \
    | grep -E "TESTS=|FAIL|Error|error:" || true
  python3 - "$name" <<'PY'
import sys, xml.etree.ElementTree as ET
r = ET.parse("results.xml").getroot()
bad = [c.get("name") for c in r.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
n = len(list(r.iter("testcase")))
print(f"{sys.argv[1]}: {n - len(bad)}/{n} passed" + (f"  FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
PY
}

run pe_mm1            pe             test_pe
A_W=9 A_SIGNED=0 ACC_W=18 run pe_mm2 pe test_pe $MM2
run array_mm1         systolic_array test_systolic_array
A_W=9 A_SIGNED=0 ACC_W=18 run array_mm2 systolic_array test_systolic_array $MM2
run mm_engine_mm1     mm_engine      test_mm_engine
A_W=9 A_SIGNED=0 ACC_W=18 run mm_engine_mm2 mm_engine test_mm_engine $MM2
run row_max           row_max        test_row_max
run scale_mask        scale_mask     test_scale_mask
run exp2              exp2_unit      test_exp2
run recip             recip_unit     test_recip
run online_softmax    online_softmax test_online_softmax
run o_accumulator     o_accumulator  test_o_accumulator
run normalize         normalize      test_normalize
run attn_tile         attn_tile      test_attn_tile
echo "ALL UNIT SUITES PASSED"
