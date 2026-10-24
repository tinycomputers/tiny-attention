#!/usr/bin/env bash
# Phase 0 toolchain spike: Verilate counter.sv once -> one WASM module that exposes a CounterSim
# class; JS creates as many independent instances as it likes.
# Requires: verilator >= 5.036, Emscripten (em++ on PATH). `source scripts/env.sh` first.
# Outputs: web/public/spike/counter.mjs + counter.wasm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OBJ_DIR="$SCRIPT_DIR/obj_dir"
OUT_DIR="$PROJECT_ROOT/web/public/spike"
SRC="$PROJECT_ROOT/rtl/spike/counter.sv"

VROOT="$(verilator --getenv VERILATOR_ROOT 2>/dev/null || true)"
if [ -z "$VROOT" ] || [ ! -f "$VROOT/include/verilated.h" ]; then
  for c in /usr/local/share/verilator /usr/share/verilator; do
    [ -f "$c/include/verilated.h" ] && VROOT="$c" && break
  done
fi
echo "[info] VERILATOR_ROOT = $VROOT"

echo "[1/2] Verilating counter.sv..."
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"
verilator --cc "$SRC" --top-module counter --prefix Vcounter --Mdir "$OBJ_DIR" -Wall

echo "[2/2] Compiling to WASM..."
# verilated_threads.cpp needs pthread affinity APIs Emscripten lacks; wasm/emscripten_stubs.cpp stubs them.
mkdir -p "$OUT_DIR"
em++ -O2 -std=c++17 \
  -DVL_IGNORE_UNKNOWN_ARCH \
  -I "$OBJ_DIR" -I "$VROOT/include" \
  "$SCRIPT_DIR/harness.cpp" \
  "$OBJ_DIR"/Vcounter*.cpp \
  "$VROOT/include/verilated.cpp" \
  "$VROOT/include/verilated_threads.cpp" \
  "$SCRIPT_DIR/../emscripten_stubs.cpp" \
  -lembind \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORT_NAME=createTinyAttnSpike \
  -o "$OUT_DIR/counter.mjs"

echo "Done: $OUT_DIR/counter.mjs + counter.wasm ($(du -h "$OUT_DIR/counter.wasm" | cut -f1))"
