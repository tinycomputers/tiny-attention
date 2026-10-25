# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

**Always read `CONTEXT.md` before any task.** The full roadmap is in `docs/PLAN.md`.

## Project Summary

TinyAttention is a real, synthesizable FlashAttention engine in SystemVerilog (online softmax, two 4×4 INT8
systolic engines, NAIVE and FLASH schedules), compiled to WASM via Verilator + Emscripten and visualized live
in the browser.

Pipeline: `rtl/*.sv` -> Verilator C++ -> Emscripten -> `web/public/tiny_attn.wasm` -> React island -> SVG.

## Commands (toolchain copied from tiny-cpu)

Toolchains are project-local (no root needed). Bootstrap once, then source the env in every shell:

```bash
bash .tools/setup.sh && bash .tools/emsdk.sh      # Verilator 5.036 + Emscripten
python3 -m venv .venv && .venv/bin/pip install -r sim/requirements.txt
source scripts/env.sh
```

Never export `VERILATOR_ROOT`.

### Golden model (Phase 1)
```bash
pytest sim/golden -q                  # SEEDS=1000 for the full sweep
python sim/golden/gen_luts.py     # regenerates rtl/lut_pkg.sv; commit the result
python sim/golden/study.py 200    # numerics study (see docs/NUMERICS.md)
```

### RTL lint
```bash
bash sim/lint.sh      # -Wall on every module, plus the MM2 parameter set
```

### Simulation (cocotb)
```bash
bash sim/run_unit.sh   # every unit suite, one SIM_BUILD each; exp2/recip exhaustive (~2 min)
cd sim && make MODULE=test_counter TOPLEVEL=counter VERILOG_SOURCES="$PWD/../rtl/spike/counter.sv" WAVES=0   # Phase 0 spike
```

### WASM
```bash
bash wasm/spike/build.sh      # Phase 0 toolchain spike -> web/public/spike/
node wasm/spike/test.mjs      # headless check of the spike build
```

### Frontend (from `web/`)
```bash
pnpm dev | build | lint | typecheck | format
```

## Branching
`main` is protected: work on `feat/*`, `fix/*`, `docs/*` branches and open PRs.
