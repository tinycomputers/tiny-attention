#!/usr/bin/env bash
# Source me: `source scripts/env.sh`
# Puts the project-local toolchain (.tools/, .venv/) on PATH. Nothing here needs root.
# Bootstrap once: bash .tools/setup.sh (Verilator) and bash .tools/emsdk.sh (Emscripten),
# then `python3 -m venv .venv && .venv/bin/pip install -r sim/requirements.txt`.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset VERILATOR_ROOT   # Verilator derives it from its own location; setting it breaks it
export PATH="$_ROOT/.tools/verilator-install/bin:$_ROOT/.venv/bin:$PATH"
[ -f "$_ROOT/.tools/emsdk/emsdk_env.sh" ] && source "$_ROOT/.tools/emsdk/emsdk_env.sh" >/dev/null 2>&1
unset _ROOT
