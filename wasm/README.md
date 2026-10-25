# wasm/

- `build.sh`: Verilates `rtl/` (top `tiny_attn_top`) and links the engine into `web/public/tiny_attn.{mjs,wasm}`.
  Output: 206 KB raw, 81 KB gzipped (budget 500 KB gzipped, enforced by `wasm.yml`).
- `harness/`: `TinyAttnSim` (one model instance with its own Verilator context) and the embind bindings.
- `test.mjs`: Node test. Runs the golden fixtures in `fixtures/` with two or more instances interleaved, checks O, bytes,
  tiles, K/V loads and cycles exactly, and checks that a fresh instance reproduces a run.
- `fixtures/`: golden runs exported by `sim/golden/export_fixtures.py`.
- `spike/`: the Phase 0 toolchain spike (two counters in one module), kept as a regression check.

```bash
source scripts/env.sh
bash wasm/build.sh && node wasm/test.mjs
```

The JavaScript API is in `docs/STATE_SCHEMA.md`.
