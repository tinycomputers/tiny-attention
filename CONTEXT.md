# CONTEXT.md - TinyAttention

Read this before any task. Full plan: `docs/PLAN.md`.

## What this is
A real, synthesizable FlashAttention engine in SystemVerilog: two 4×4 INT8 systolic matmul engines, an
online-softmax unit (`m`, `l`, base-2 `exp2`), an O accumulator that rescales, and a reciprocal normalizer. It is
compiled to WASM with Verilator + Emscripten and visualized live in the browser. It is the fourth project after
tiny-gpu, tiny-tpu and tiny-cpu, and reuses tiny-cpu's toolchain. Nothing on screen is faked: every PE value,
running max, rescale factor, SRAM tile and HBM byte comes from a `dbg_*` port on the RTL.

## Status
- Planning: DONE (`docs/PLAN.md`).
- Phase 0 (scaffold + toolchain spike): DONE. Verified locally: Verilator lint, cocotb (4 tests), WASM spike with two
  instances of one model (Node + browser, ticking independently), web lint/typecheck/build.
- Phase 1 (golden models + numerics): DONE. `pytest sim/golden -q` (52 tests; `SEEDS=1000` for the full sweep) green.
  Formats and rounding frozen in `docs/ARCHITECTURE.md`; the study that chose them is `docs/NUMERICS.md`.
- Phase 2 (RTL arithmetic blocks): DONE. `pe, systolic_array, mm_engine, row_max, scale_mask, exp2_unit, recip_unit`,
  `-Wall` clean (`bash sim/lint.sh`), all cocotb suites green (`bash sim/run_unit.sh`, ~2 min).
- Phase 3 (one tile end to end): DONE. `online_softmax, o_accumulator, normalize, attn_tile` (MM1 -> softmax -> MM2 -> O,
  then normalize); lockstep vs `fixed.flash(trace=True)` at every tile boundary; tile = 20 cycles `tile_start` -> `tile_done`.


## Local toolchain note
`.tools/verilator-install` and `.tools/emsdk` are symlinks to tiny-cpu's builds (gitignored) to avoid recompiling.
On a fresh machine delete the symlinks and run `.tools/setup.sh` and `.tools/emsdk.sh` instead.

- Phase 4 (memory + FLASH controller + top): DONE. `sram_bank, hbm_model, dma, flash_ctrl, perf_counters, tiny_attn_top`
  and the 32-row attn_tile. End to end: bit-exact O, HBM bytes, tiles, K/V loads and busy cycles all equal the golden /
  closed-form models for every N, causal, cfg_qrows and latency tested (`SEEDS=200 bash sim/run_e2e.sh`).
- Phase 5 (overlap, NAIVE, perf pins): DONE. `naive_ctrl` (4 passes, block-by-block), FLASH double-buffered K/V prefetch
  (`cfg_overlap`), tile operations OP_NZ / OP_NM / OP_NP / OP_NPV, HBM 2048 words. Full e2e (200 seeds) and unit suites
  pass; perf baseline in `sim/perf_baseline.json` (re-record with `PERF_RECORD=1 COCOTB_TEST_FILTER=perf_grid bash sim/run_e2e.sh`).
- Phase 6 (WASM harness): DONE. `wasm/build.sh` builds `web/public/tiny_attn.{mjs,wasm}` (81 KB gzipped); `node wasm/test.mjs`
  checks three golden fixtures on interleaved instances, exactly. JS API in `docs/STATE_SCHEMA.md`.
- Phase 7 (web visualizer `/app`): DONE. Presets, editable Q/K/V (quantized in the browser), schedule, tile values, K/V
  buffers (read from the RTL), HBM meter against the golden traffic model, heatmap (engine P for NAIVE, float reference for
  FLASH), Playwright smoke test (5 tests, production build).
- Phase 8 (L1 softmax cell, L2 tile dataflow, L4 naive vs flash, N sweep, error panel): DONE. Engine readouts `tileDone`
  and `getDatapath`. 12 Playwright tests, 15 Vitest, Node WASM test, one-seed cocotb. Light-mode screenshots not reviewed.
- Phase 9 (docs, landing page, launch): built. Landing page with the animated FLASH schedule, six docs pages with live
  demos, `/og` social card, generated `measured.ts`, README rewritten for the series. Deployment: see the PR and
  ARCHITECTURE §14.
- Phase 10 (FPGA, ECP5 LFE5U-85F): partly done. UART wrapper (`fpga/ecp5/`), simulated over the UART end to end, synthesis
  with no latches, utilization and an fmax estimate (11.5 MHz; not closed at 25 MHz). Not done: pin constraints for the
  ULX3S, a board run, timing closure, block RAM for the memories (combinational reads map to LUT RAM).
- Next: the Phase 10 follow-ups above, or Phase 9's deployment (blocked: the production deploy was denied by the auto-mode classifier).

## Gotchas found in Phase 10
- Yosys rejects: unpacked array localparams, package imports in a module header, part-selects of package parameters,
  width casts with expression widths, and compound async resets. All are fixed in rtl/ (see ARCHITECTURE §15).
- YoWASP tools only read and write the working directory: outputs go in `fpga/ecp5/build`, inside the project.
- The installed Verilator leaves `CFG_CXXFLAGS_PCH_I` empty, so a freshly generated model fails to link. `sim/run_fpga.sh`
  passes `-include` through MAKEFLAGS. Other suites did not hit this.
- The wrapper's netlist is ~48 MB as JSON. `fpga.yml` deletes it after the run.

## Gotchas found in Phase 8
- The debug datapath wires are idle between tiles. The L2 view keeps the last busy clock's values, or the panel shows zeros.
- A datapath or buffer value is only meaningful while the engine is inside a tile; say so in the UI.
- Verilator's wide ports are word arrays; the `wordsOf` template reads any of them.
- A full run in the browser is cheap, but the pages overflowed at 375 px only after a run (long decimals in the output grid,
  then a 1 px overhang in the tile grids). The phone check must run a computation first.
- `astro preview` hung (process state D) during Phase 8; a static file server over `dist/` works. Restart it after each build,
  because the build removes its working directory.

## Gotchas found in Phase 7
- `wasm/fixtures/` holds engine runs and `quantize_cases.json`; the Node test must skip the quantization file (it did not for
  a while and aborted in configure with no useful message).
- Embind gives 64-bit integers to JavaScript as BigInt, which breaks JSON comparison. Tile values are 32-bit for that reason.
- The tile core's debug S/Z/P registers are only written by FLASH. NAIVE views must not show them.
- Playwright 1.64 needs its own Chromium build (`pnpm exec playwright install chromium`); the cache has other versions.
- `astro dev` hung without output in this environment; `astro build` + `astro preview` works for browser checks.

## Gotchas found in Phase 5
- The NAIVE tile operations reuse the online-softmax unit: OP_NM uses its m path, OP_NP uses its P and l path with alpha = 1
  (exp2(0) = 32768), OP_NPV uses o_accumulator with alpha forced to 32768. Golden equality holds, but that identity is the thing to re-check first if NAIVE ever breaks.
- A wrong alpha in the NAIVE O update fails the e2e test (checked by injecting it).

## Gotchas found in Phase 4
- Disk: the root filesystem was 100% full during Phase 4. The Bash tool writes its output to /tmp, so with no space every
  command failed, even `true`. Build dirs (sim/sim_build) are deleted after each suite in run_unit.sh / run_e2e.sh for this reason.
- Config ports take tile counts (cfg_nb = N/4, cfg_gg = qrows/4), not raw values, so no unused low bits reach lint.
- DMA word index is 6 bits (bursts are at most 64 words); staging reads use index[5:1].
- The first plan's per-tile cycle estimate (2L+32) was wrong. The real schedule is in ARCHITECTURE.md section 9.3.

## Gotchas found in Phase 3
- `fixed.flash` calls `online_softmax_step` and `o_update`; the RTL modules are tested against those same functions, so the
  golden has no duplicate arithmetic to drift.
- `attn_tile` has no overlap yet: weights load (4 cycles) then 20 cycles per tile. Do not add overlap before the controller exists.
- The tile core takes K as Kᵀ rows (a strided SRAM read in Phase 4) and V as plain rows.

## Gotchas found in Phase 2
- cocotb: pass Verilator `-G` parameter overrides via `EXTRA_ARGS`. A command-line `COMPILE_ARGS=...` replaces cocotb's own
  and breaks the build. Packages (`lut_pkg.sv`, `tiny_attn_pkg.sv`) must come first in `VERILOG_SOURCES`.
- Tests take parameters from env vars (`A_W`, `A_SIGNED`, `ACC_W`) matching the `-G` overrides; `sim/run_unit.sh` sets both.
- Cycle convention for sequential tests: drive 1 ns after a rising edge, sample at the falling edge (`sim/tbutil.py`).
- A mutation that changes nothing (e.g. a delay count in a generate branch that is never taken) proves nothing; check the
  mutant actually differs before trusting that a suite catches it.

## Gotchas inherited from tiny-cpu Phase 0
- Emscripten lacks pthread affinity APIs; `verilated_threads.cpp` is still required, so link `wasm/emscripten_stubs.cpp`.
- Verilator `--Mdir` does not create parent dirs: `mkdir -p` first.
- cocotb 2.x needs Verilator >= 5.036 and `Clock(..., unit="ns")`.
- `.tools/setup.sh` patches Verilator's empty `PACKAGE_STRING` for cocotb. Never export `VERILATOR_ROOT`.

## Fixed decisions (do not re-litigate; change only via docs/ARCHITECTURE.md)
| Decision | Choice |
|---|---|
| Scope | Single head, forward only, `D = 4`, tiles 4×4, `NMAX = 32` (runtime `cfg_n` multiple of 4) |
| Schedules | `cfg_mode`: FLASH (FA-2 order, Q blocks outer) and NAIVE (S/P materialized in HBM). Same datapath |
| Softmax | Base 2; `log₂e/√d` and quant scales folded into host `cfg_scale` (24-bit UQ4.20). Deferred normalization in both modes |
| Formats | Q/K/V INT8; S INT17; `cfg_scale` UQ4.20 (24 b); Z Q8.8 sat (mask = `0x8000`); α/exp2 UQ1.15; **P UQ1.8 (9 b, 256 = 1.0)**; l 18 b (×2^12); O 26 b signed (×2^4); out Q8.8 in V-LSB |
| Rounding | Truncate everywhere except `P` (UQ1.15→UQ1.8) and the final `out`, which round to nearest |
| exp2 | `LUT256[frac] >> −int`, 0 when shift ≥ 16. recip: LZC + LUT64 + 1 Newton-Raphson step |
| LUTs | Generated by `sim/golden/gen_luts.py` → `rtl/lut_pkg.sv`; one source for RTL and golden |
| FLASH vs NAIVE | Not bit-identical; FLASH is the *more* accurate (P quantized vs running max). Pinned and documented |
| `cfg_qrows` | 4/8/16/32 rows per resident Q block (FLASH). K/V loaded once per block; output independent of it |
| Memory | HBM model: 1 word/cycle, `cfg_hbm_latency` 1…64. On-chip SRAM: Q, K×2, V×2, Z, P, O tiles |
| Knobs | `cfg_causal` (skip tiles above diagonal), `cfg_overlap_en` (ping-pong K/V prefetch) |
| Verification | Unit (exhaustive exp2/recip) + tile-boundary lockstep + e2e random bit-exact + float accuracy bounds + pinned cycles/HBM bytes |
| WASM | One model; each `TinyAttnSim` owns its own `VerilatedContext` (L4 runs two instances) |
| FPGA | ECP5 (ULX3S), not iCE40 (32 MACs need DSPs) |

## Conventions
- SystemVerilog: 4-space indent, `logic`, `always_ff`/`always_comb`, `-Wall` lint clean.
- Python: 4-space indent, tests under `sim/`, fixed seed 42 (see `sim/conftest.py`).
- Every rounding step in RTL has a matching line in `sim/golden/fixed.py` and an entry in `docs/ARCHITECTURE.md`.
- Web: Astro + React islands + Tailwind, TypeScript strict. WASM is loaded only inside
  `client:only="react"` islands, never during SSR.
- `docs/STATE_SCHEMA.md` and `web/src/lib/state-schema.ts` must stay in sync (from Phase 6).
- Raw fixed-point integers cross the WASM boundary; conversion to reals happens in `web/src/lib/formats.ts`.
