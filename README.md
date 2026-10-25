# TinyAttention

A real, synthesizable **FlashAttention engine** in SystemVerilog, compiled to WebAssembly and running live
in your browser: online softmax with a running max and sum, SRAM tiles filling and emptying, and standard
vs FlashAttention memory traffic side by side.

Fourth project in the series after [tiny-gpu](https://github.com/deaneeth/tiny-gpu),
[tiny-tpu](https://github.com/deaneeth/tiny-tpu) and tiny-cpu. Same rule as the others: nothing on screen is
faked; every value comes from a `dbg_*` port on the compiled RTL.

**Status:** Phases 0-6 done: the engine (FLASH, NAIVE, K/V overlap) is bit-exact against the golden model and compiled to WebAssembly (81 KB gzipped), with a Node test over golden fixtures. Next is the web UI. See [`docs/PLAN.md`](docs/PLAN.md) for the roadmap.

## Quick start

```bash
bash .tools/setup.sh && bash .tools/emsdk.sh      # Verilator 5.036 + Emscripten (project-local)
python3 -m venv .venv && .venv/bin/pip install -r sim/requirements.txt
source scripts/env.sh

bash wasm/build.sh && node wasm/test.mjs              # engine WASM + golden fixtures
bash wasm/spike/build.sh && node wasm/spike/test.mjs   # toolchain spike
cd web && pnpm install && pnpm dev                     # open /spike
```

## License
MIT
