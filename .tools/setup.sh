#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x verilator-install/bin/verilator ]; then
  rm -rf verilator-src && git clone -q --depth 1 --branch v5.036 https://github.com/verilator/verilator.git verilator-src
  # CMake builds leave PACKAGE_STRING empty, which cocotb needs to parse the version
  sed -i 's|#define PACKAGE_STRING ""|#define PACKAGE_STRING "Verilator 5.036 2025-04-27"|' verilator-src/src/config_package.h.in
  cmake -S verilator-src -B verilator-src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/verilator-install" >/dev/null
  cmake --build verilator-src/build -j8 >/dev/null
  cmake --install verilator-src/build >/dev/null
fi
echo VERILATOR_DONE
