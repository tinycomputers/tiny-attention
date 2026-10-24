# wasm/

`spike/` is the Phase 0 proof that Verilator + Emscripten work here, including **two independent
instances of one Verilated model** (each with its own `VerilatedContext`) in one module. Phase 6 adds
`harness.cpp`, `bindings.cpp`, `build.sh`, `test.mjs`.

```bash
source scripts/env.sh
bash wasm/spike/build.sh && node wasm/spike/test.mjs
```
