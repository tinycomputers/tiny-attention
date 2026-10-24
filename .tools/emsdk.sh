#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -d emsdk ] || git clone -q --depth 1 https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
echo EMSDK_DONE
