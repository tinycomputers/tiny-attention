# TinyAttention: JavaScript state schema (Phase 6)

What the WebAssembly module exposes per clock, and how to drive it. The schema mirrors the C++ structs in
`wasm/harness/harness.h`; if one changes, the other must change in the same commit.

Phase 6 exports a **subset** of the planned per-cycle state. The full `CycleState` (per-PE values, softmax lanes, SRAM
tags, tile matrices) needs wide-signal access through Verilator and is the first item of Phase 7. The RTL already has
the ports for it (`dbg_*` in `tiny_attn_top` and `attn_tile`); only the bindings are missing.

## Snapshot (returned by `step()`, `snapshot()`)

| Field | Type | Meaning |
|---|---|---|
| `cycle` | number | clock edges since reset |
| `ctrlState` | number | `dbg_ctrl_state`: the active controller's state enum (`flash_ctrl` or `naive_ctrl`, see their source) |
| `busy` | boolean | controller not idle |
| `done` | boolean | one-cycle pulse: the DONE state |
| `hbmRdBytes` | number | DMA read bytes since start (4 per word) |
| `hbmWrBytes` | number | DMA write bytes since start |
| `cycles` | number | busy-cycle counter (state != idle, DONE included) |
| `tiles` | number | tile computations (FLASH: group tiles; NAIVE: pass-1 tile pairs) |
| `kvLoads` | number | K/V tile loads (FLASH: K loads incl. prefetch; NAIVE: one per tile pair) |

**Counters are final one edge after `done`.** `done` is high during the DONE cycle, and the busy counter increments at the
end of that cycle. `wasm/test.mjs` takes one more `step()` before reading.

## Methods

```ts
class TinyAttnSim {
  reset(): void;
  configure(n: number, qrows: number, causal: boolean, mode: 0 | 1, overlap: boolean,
            latency: number, scale: number): void;  // mode 0 = FLASH, 1 = NAIVE; scale < 2^24
  loadQKV(q: number[], k: number[], v: number[]): void;  // n*4 int8 values each, row-major; n*3 cycles
  start(): void;                  // one-cycle start pulse; the next step() is the first cycle of the run
  step(): Snapshot;               // one clock
  runToDone(maxCycles: number): number;  // edges taken until done, -1 on timeout
  readOutput(): number[];         // n*4 int16 values (O, Q8.8 V-LSB), row-major
  readHbm(addr: number, len: number): number[];  // raw 32-bit words (Z/P tiles are at 256 and 768)
  snapshot(): Snapshot;
  isDone(): boolean;
  delete(): void;
}
```

Input validation throws: `n` must be a multiple of 4 in 4..32, `qrows` one of 4, 8, 16, 32 and not above `n`, latency 0..64,
`scale` below 2^24.

## Run sequence

1. `new mod.TinyAttnSim()`, then `configure(...)`.
2. `loadQKV(q, k, v)`: quantized int8 values from `sim/golden/fixed.py` (`quantize`), and `scale = cfgScale`.
3. `start()`, then `step()` until `done`, or `runToDone(max)`.
4. `step()` once more, then `readOutput()`.

Multiple instances are independent: each owns its Verilator context. Steps may be interleaved across instances.

## Address map of the HBM model (words)

| Region | Base | Size |
|---|---|---|
| Q | 0 | 32 rows (one word each) |
| K | 32 | 32 rows |
| V | 64 | 32 rows |
| O | 96 | 64 words (two per row) |
| Z tiles (NAIVE) | 256 | 8 words per tile, tile (qi, kj) at `256 + 64·qi + 8·kj` |
| P tiles (NAIVE) | 768 | same layout |

## Fixtures

`wasm/fixtures/*.json` are exported by `sim/golden/export_fixtures.py` and checked for freshness by `pytest sim/golden`.
Each holds the quantized inputs, the expected O, bytes, tiles, K/V loads and cycles.
