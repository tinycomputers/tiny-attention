#!/usr/bin/env bash
# Build the TinyAttn engine as a WebAssembly module: Verilate rtl/ (top: tiny_attn_top), compile the C++ harness
# and the Verilator runtime with Emscripten, link into web/public/tiny_attn.{mjs,wasm}.
# Requires: verilator >= 5.036 and Emscripten (`source scripts/env.sh` first).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="$ROOT/wasm/build/obj"
OUT="$ROOT/web/public"

VROOT="$(verilator --getenv VERILATOR_ROOT)"
[ -f "$VROOT/include/verilated.h" ] || { echo "VERILATOR_ROOT not found: $VROOT" >&2; exit 1; }

RTL_TOP=tiny_attn_top
SRCS=("$ROOT/rtl/lut_pkg.sv" "$ROOT/rtl/tiny_attn_pkg.sv")
for f in "$ROOT"/rtl/*.sv; do
    case "$f" in */lut_pkg.sv|*/tiny_attn_pkg.sv) ;; *) SRCS+=("$f") ;; esac
done

echo "[1/2] Verilating ${RTL_TOP}..."
rm -rf "$OBJ"
mkdir -p "$OBJ"
verilator --cc "${SRCS[@]}" --top-module "$RTL_TOP" --prefix Vtiny_attn_top --Mdir "$OBJ" \
    -O3 -Wno-fatal -Wno-UNOPTFLAT >"$OBJ/verilator.log" 2>&1 || { tail -40 "$OBJ/verilator.log"; exit 1; }

echo "[2/2] Compiling to WASM..."
mkdir -p "$OUT"
em++ -O2 -std=c++17 -DVL_IGNORE_UNKNOWN_ARCH \
    -I "$OBJ" -I "$VROOT/include" -I "$VROOT/include/vltstd" -I "$ROOT/wasm/harness" \
    "$ROOT/wasm/harness/harness.cpp" "$ROOT/wasm/harness/bindings.cpp" \
    "$OBJ"/Vtiny_attn_top*.cpp \
    "$VROOT/include/verilated.cpp" \
    "$VROOT/include/verilated_threads.cpp" \
    "$ROOT/wasm/emscripten_stubs.cpp" \
    -lembind \
    -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORT_NAME=createTinyAttn -s ENVIRONMENT=web,node \
    -o "$OUT/tiny_attn.mjs"

WASM="$OUT/tiny_attn.wasm"
RAW=$(stat -c %s "$WASM")
GZ=$(gzip -9 -c "$WASM" | wc -c)
echo "Done: $OUT/tiny_attn.mjs + tiny_attn.wasm  (raw $RAW bytes, gzip $GZ bytes)"
