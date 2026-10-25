#!/usr/bin/env bash
# Lint every RTL top with -Wall (warnings are errors). Packages first, then the module under test.
# Run from anywhere: bash sim/lint.sh
set -euo pipefail
cd "$(dirname "$0")/../rtl"
PKGS="lut_pkg.sv tiny_attn_pkg.sv"
ALL="$PKGS pe.sv systolic_array.sv mm_engine.sv row_max.sv scale_mask.sv exp2_unit.sv recip_unit.sv online_softmax.sv o_accumulator.sv normalize.sv attn_tile.sv"
for top in pe systolic_array mm_engine row_max scale_mask exp2_unit recip_unit online_softmax o_accumulator normalize attn_tile; do
  echo "lint: $top"
  verilator --lint-only -Wall --top-module "$top" $ALL
done
# the MM2 configuration (9-bit unsigned P x int8 V -> 18-bit psum) must lint too
echo "lint: mm_engine (MM2 params)"
verilator --lint-only -Wall --top-module mm_engine -GA_W=9 -GA_SIGNED=0 -GACC_W=18 $ALL
echo "lint: OK"
