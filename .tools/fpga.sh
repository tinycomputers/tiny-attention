#!/usr/bin/env bash
# FPGA tools for Phase 10: Yosys and nextpnr for ECP5, as YoWASP wheels (WebAssembly builds, no system packages).
# Installs into .tools/fpga-venv. Usage: bash .tools/fpga.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x fpga-venv/bin/yowasp-yosys ] || [ ! -x fpga-venv/bin/yowasp-nextpnr-ecp5 ]; then
  python3 -m venv fpga-venv
  fpga-venv/bin/pip install -q yowasp-yosys yowasp-nextpnr-ecp5
fi
echo FPGA_TOOLS_DONE
