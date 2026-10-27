<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="TinyAttention" width="470" />
  </picture>

  <h1>FlashAttention in SystemVerilog</h1>

  <p>
    <strong>A synthesizable FlashAttention engine, verified bit-exact against a golden model,<br>
    compiled to WebAssembly, and running live in your browser.</strong>
  </p>

  <p>
    The tiled online-softmax schedule from the FlashAttention paper, as real RTL: the N × N score matrix never leaves the chip.
    The same RTL runs in the browser and on an ECP5 FPGA.
  </p>

  <p>
    <a href="https://github.com/tinycomputers/tiny-attention/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT License" /></a>
    <img src="https://img.shields.io/badge/RTL-SystemVerilog-7c3aed?style=flat-square" alt="RTL: SystemVerilog" />
    <img src="https://img.shields.io/badge/Runtime-WebAssembly-654ff0?style=flat-square&logo=webassembly&logoColor=white" alt="Runtime: WebAssembly" />
    <img src="https://img.shields.io/badge/Verified-cocotb%20%2B%20golden%20model-3b82f6?style=flat-square" alt="Verified: cocotb and a golden model" />
    <img src="https://img.shields.io/badge/FPGA-ECP5%20LFE5U--85F-f59e0b?style=flat-square" alt="FPGA: ECP5 LFE5U-85F" />
    <img src="https://img.shields.io/badge/Frontend-Astro%20%2B%20React-ff5d01?style=flat-square&logo=astro&logoColor=white" alt="Frontend: Astro and React" />
    <img src="https://img.shields.io/badge/TypeScript-strict-3178c6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript strict" />
  </p>

  <p>
    <a href="https://github.com/tinycomputers/tiny-attention/actions/workflows/rtl.yml"><img src="https://github.com/tinycomputers/tiny-attention/actions/workflows/rtl.yml/badge.svg" alt="RTL CI" /></a>
    <a href="https://github.com/tinycomputers/tiny-attention/actions/workflows/wasm.yml"><img src="https://github.com/tinycomputers/tiny-attention/actions/workflows/wasm.yml/badge.svg" alt="WASM Build" /></a>
    <a href="https://github.com/tinycomputers/tiny-attention/actions/workflows/web.yml"><img src="https://github.com/tinycomputers/tiny-attention/actions/workflows/web.yml/badge.svg" alt="Web CI" /></a>
    <a href="https://github.com/tinycomputers/tiny-attention/actions/workflows/fpga.yml"><img src="https://github.com/tinycomputers/tiny-attention/actions/workflows/fpga.yml/badge.svg" alt="FPGA synthesis" /></a>
  </p>

  <p>
    <a href="https://tiny-attention.vercel.app"><img src="https://img.shields.io/badge/Open%20Live%20Visualizer-58a6ff?style=for-the-badge&logoColor=black" alt="Open Live Visualizer" /></a>
    &nbsp;
    <a href="https://tiny-attention.vercel.app/docs/flashattention"><img src="https://img.shields.io/badge/Read%20the%20Docs-3fb950?style=for-the-badge" alt="Read the Docs" /></a>
  </p>
</div>

---

## Preview

<div align="center">

![The TinyAttention visualizer after one FLASH pass: schedule, tile values, HBM traffic and the output](docs/media/visualizer.png)

</div>

Pick a preset (a toy sentence, a previous-token head, an attention sink, and more), press **Start**, and step the clock.
Every tile, buffer and byte in the screenshot is read from the compiled RTL running in your browser.

---

## What is FlashAttention?

Attention computes `softmax(Q Kᵀ / √d) V`. Standard implementations build the full `N × N` score matrix in memory,
read it back for the softmax, and read it again for the product with `V`. At long sequence lengths that matrix dominates
the memory traffic, and that traffic is what limits speed.

**FlashAttention** computes the same output one tile at a time. A block of query rows stays on chip, key and value tiles
stream past it, and an **online softmax** keeps a running maximum `m` and running sum `l` for each row. When a new tile
raises the maximum, the old sum and the output accumulator are rescaled by `α = exp(m_old − m_new)`. The score matrix is never
written to memory, and the result is exact.

This project implements that schedule in hardware, along with the standard three-pass schedule (NAIVE) for comparison.
Both run on the same arithmetic units, so the difference you see is the schedule, not the datapath.

---

## Highlights

- **A real FlashAttention datapath in RTL.** Two 4 × 4 systolic arrays (`S = Q·Kᵀ` and `P·V`), scale and causal mask,
  online softmax with an exp2 unit and a reciprocal, and an O accumulator that rescales by α. All of it is in `rtl/`.
- **Bit-exact against a golden model.** The engine's output, HBM byte counts, tile counts and K/V loads match the Python
  golden model on every end-to-end run, and busy cycles match a closed-form schedule model.
- **Fixed point, end to end.** INT8 Q, K and V with per-tensor scales folded into one 24-bit `cfg_scale`; UQ1.15 exp2 and
  reciprocal from packed lookup tables; every width is documented in [ARCHITECTURE](docs/ARCHITECTURE.md).
- **Runs in the browser.** Verilator compiles the RTL to C++, Emscripten compiles that to a 84 KB (gzip) WebAssembly module,
  and the visualizer reads every value from its debug ports. Nothing is reimplemented in JavaScript.
- **Synthesizes for an FPGA.** ECP5 (LFE5U-85F, the ULX3S chip) with no inferred latches, behind a UART command interface.
- **Teaching visualizer.** Four views: the softmax cell, the tile datapath, the tile schedule, and a naive-versus-flash
  comparison with an N sweep and an error panel.

---

## How It Works

```
 rtl/*.sv ──► Verilator ──► C++ model ──► Emscripten ──► tiny_attn.wasm ──► Astro + React visualizer
 (the design)  (cycle-accurate)            (wasm/build.sh)   (web/public)        (reads debug ports)
      │
      ├── cocotb unit suites and end-to-end tests against sim/golden (Python, bit-exact)
      ├── Verilator lint with -Wall, on every module and the full design
      └── Yosys + nextpnr for ECP5 (fpga/ecp5)
```

The browser never computes attention itself. It steps the compiled hardware one clock at a time and draws what the
hardware reports. If the RTL is wrong, the visualizer shows the wrong thing, and the test suites catch it first.

---

## Hardware

**Tile core (`attn_tile.sv`).** One 4-row group against one 4-row K/V tile per step, with 20 cycles from tile start to
tile done. MM1 computes `S` on a 4 × 4 weight-stationary array, the scale and causal mask apply, the online softmax updates
`m`, `l` and `α`, and MM2 computes `P·V` on a second array. The O accumulator rescales by `α` and adds the new product.
Five operations share this datapath: the FLASH step, and the four NAIVE passes.

**Memory and control.** `hbm_model` is a 2,048-word memory with byte counters. `dma` moves bursts with a configurable
first-word latency (0 to 64 cycles). K and V are double-buffered when overlap is on. `flash_ctrl` loops over Q blocks and
K/V tiles, skips tiles above the causal diagonal, and normalizes and stores each block. `naive_ctrl` runs the three-pass
softmax through HBM.

**Formats.** INT8 inputs, 17-bit `S`, Q8.8 log-domain `Z` and `m`, UQ1.15 exp2 and `α`, UQ1.8 `P`, 18-bit `l`, 26-bit `O`,
and a Q8.8 output. The full table, with the reasoning, is in the [fixed-point docs](https://tiny-attention.vercel.app/docs/fixed-point).

### Schedule and traffic

At `N = 32`, non-causal, the engine moves these bytes (from the golden traffic model, checked against the RTL counters):

| Schedule | Q block | HBM bytes | vs NAIVE |
|:---|:---:|---:|:---:|
| NAIVE (three-pass, scores in HBM) | 4 rows | 12,672 | 1× |
| FLASH | 4 rows | 2,432 | 5.2× less |
| FLASH | 16 rows | 896 | 14.1× less |

With the causal mask, the same schedules move 7,296 bytes (NAIVE), 1,536 (FLASH, 4 rows) and 768 (FLASH, 16 rows).
A larger Q block means fewer K/V loads: the output does not depend on the block size, only the traffic does.

### Cycles

Busy cycles at `N = 32`, non-causal, 10 cycles of HBM latency (`sim/perf_baseline.json`):

| Configuration | Busy cycles |
|:---|---:|
| FLASH, 4-row Q block, no overlap | 4,113 |
| FLASH, 4-row Q block, K/V overlap | 2,937 |
| FLASH, 16-row Q block, no overlap | 2,133 |
| FLASH, 16-row Q block, K/V overlap | 1,657 |
| NAIVE | 10,961 |

The sequential FLASH numbers match a closed-form schedule model exactly. The overlap and NAIVE numbers are pinned from the
RTL as regression baselines: a change to either fails the test suite.

### Accuracy

The engine's output against float attention on the same inputs, after dequantization. The error comes from INT8 input
quantization, not the datapath:

| Preset | N | Causal | Max error |
|:---|---:|:---:|---:|
| toy sentence | 12 | yes | 0.0049 |
| previous token | 16 | yes | 0.0105 |
| attention sink | 16 | yes | 0.0253 |
| uniform | 16 | no | 0.0028 |
| self | 16 | no | 0.0260 |
| random (seeded) | 16 | no | 0.0163 |

The 1,000-seed numerics study (`sim/golden/study_results_1000seeds.txt`) separates the datapath error from the total.
At `N = 32`, the datapath error averages 0.07 of one INT8 step of V for FLASH and 0.10 for NAIVE.

---

## Verification

Every row is a test in this repository, run in GitHub Actions on each pull request.

| What | How it is checked | Scope |
|:---|:---|:---|
| Output, bytes, tiles, K/V loads and cycles | cocotb end-to-end test against the golden model | every N from 4 to 32, causal and not, every block size, both schedules, overlap on and off, random HBM latency, 200 seeds |
| Arithmetic blocks | cocotb unit suites | exhaustive exp2 and reciprocal sweeps; DMA burst timing cycle by cycle |
| Golden model | `pytest sim/golden` | 54 tests: quantization, exp2 and reciprocal, both schedules against float, traffic formulas, presets, and freshness of every generated table |
| Lint | Verilator `-Wall` | every module and the full design |
| WebAssembly | `node wasm/test.mjs` | the compiled engine on golden fixtures; the committed build must match a fresh build (CI) |
| UART interface | cocotb test of the FPGA top | the engine driven over its serial protocol, bit-exact |
| Web app | Vitest and Playwright | the quantizer and tile order against shared fixtures; every page in a real browser; the visualizer at phone width and in both colour schemes |

---

## FPGA

The engine behind a UART (`fpga/ecp5/fpga_top.sv`) synthesizes for the ECP5 LFE5U-85F (CABGA381, speed grade 6, the ULX3S
chip) with Yosys and nextpnr. `bash .tools/fpga.sh && bash fpga/ecp5/synth.sh` runs the flow; the `fpga.yml` workflow runs it in CI.

| Resource (LFE5U-85F) | Used | Available | Share |
|:---|---:|---:|---:|
| LUT4 positions | 40,416 | 83,640 | 48% |
| Flip-flops | 8,726 | 83,640 | 10% |
| DSP (MULT18X18D) | 51 | 156 | 33% |
| Block RAM | 0 | 208 | 0% |

Synthesis reports no inferred latches. The fmax estimate from nextpnr is **11.5 MHz**, below the 25 MHz board clock, so
**timing is not closed yet**. The estimate has no pin constraints. The memories read combinationally, so they map to LUT
RAM rather than block RAM. Moving them to block RAM changes the cycle model. The UART interface is verified in
simulation; the design has not run on a board yet.

---

## Quick Start

Requirements: Verilator 5.036, Python 3.12 with cocotb, Node 22 with pnpm, and Emscripten for the WebAssembly build.
The repo's `.tools/` scripts install the native and FPGA tools into the project.

```bash
git clone https://github.com/tinycomputers/tiny-attention.git && cd tiny-attention
bash .tools/setup.sh && bash .tools/emsdk.sh       # Verilator 5.036 and Emscripten (project-local)
python3 -m venv .venv && .venv/bin/pip install -r sim/requirements.txt
source scripts/env.sh

bash sim/lint.sh                                   # Verilator -Wall on every module
bash sim/run_unit.sh                               # cocotb unit suites
SEEDS=200 bash sim/run_e2e.sh                      # engine against the golden model, end to end
pytest sim/golden -q                               # golden model, fixtures and generated tables

bash wasm/build.sh && node wasm/test.mjs           # the WebAssembly engine on golden fixtures

cd web && pnpm install && pnpm build && pnpm test:e2e   # the site, production build, browser tests
```

`astro dev` can hang without output in some environments. `pnpm build` followed by any static file server over
`web/dist` works.

---

## Repository Structure

```text
tiny-attention/
│
├── rtl/                       SystemVerilog: arithmetic, tile core, memory, controllers, top level
│   ├── tiny_attn_top.sv       top level: FLASH or NAIVE, K/V overlap, debug ports
│   ├── attn_tile.sv           tile core: MM1, scale and mask, online softmax, MM2, O accumulator
│   ├── flash_ctrl.sv          FlashAttention schedule and K/V prefetch
│   ├── naive_ctrl.sv          three-pass softmax through HBM
│   └── lut_pkg.sv             exp2 and reciprocal tables (generated)
│
├── sim/                       cocotb tests, golden model, runners
│   ├── golden/                Python golden model: fixed.py, float_ref.py, traffic.py, schedule.py
│   ├── test_e2e.py            engine against the golden model, end to end
│   ├── test_fpga_uart.py      the FPGA top over its UART
│   └── perf_baseline.json     pinned busy cycles
│
├── wasm/                      Verilator and Emscripten build, C++ harness, Node test
├── fpga/ecp5/                 UART wrapper, ECP5 synthesis script
├── web/                       Astro and React site: /app visualizer, /docs pages, tests
└── docs/                      ARCHITECTURE, PLAN, STATE_SCHEMA, media
```

---

## Documentation

| Page | What it covers |
|:---|:---|
| [What is attention](https://tiny-attention.vercel.app/docs/what-is-attention) | Queries, keys, values, the score matrix, causal masks. A live float demo. |
| [Online softmax](https://tiny-attention.vercel.app/docs/online-softmax) | The running max and sum, the rescale, and the engine's row state recorded tile by tile. |
| [FlashAttention](https://tiny-attention.vercel.app/docs/flashattention) | The tile schedule, why it saves memory traffic, and the byte counts of both schedules. |
| [Fixed point](https://tiny-attention.vercel.app/docs/fixed-point) | Number formats, the exp2 unit, INT8 quantization with a live demo, and measured error. |
| [Architecture](https://tiny-attention.vercel.app/docs/architecture) | The blocks of the RTL and how they connect. |
| [Verification](https://tiny-attention.vercel.app/docs/verification) | Every check, the command that runs it, and the measured results. |

The design plan and the status of each phase are in [`docs/PLAN.md`](docs/PLAN.md). The full architecture, with the
measured schedule and traffic models, is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Roadmap

| Status | Item |
|:---:|:---|
| ✅ | RTL FlashAttention engine (FLASH, NAIVE, K/V overlap), bit-exact against the golden model |
| ✅ | WebAssembly build and browser visualizer with four views |
| ✅ | Docs site, landing page, and FPGA synthesis for ECP5 with a UART interface |
| 🔲 | Timing closure at 25 MHz on the ECP5 |
| 🔲 | Block RAM for the HBM model and the K/V buffers (changes the cycle model) |
| 🔲 | ULX3S pin constraints and a run on the board |
| 🔲 | Larger head dimension (`d` = 8, 16) by accumulating MM1 over d-chunks |
| 🔲 | Decode mode: one query row against a growing K/V cache |

---

## Related Projects

Part of the [TinyComputers](https://github.com/tinycomputers) series of small, real hardware projects, each verified against
a reference model and visualized in the browser:

- [tiny-gpu](https://github.com/tinycomputers/tiny-gpu): a SIMT GPU simulator in Python
- [tiny-tpu](https://github.com/tinycomputers/tiny-tpu): a 4 × 4 systolic array in SystemVerilog, running in the browser
- [tiny-compiler](https://github.com/tinycomputers/tiny-compiler): compiles one tensor program to tiny-tpu tile schedules or tiny-gpu kernels

---

## Contributing

Issues and pull requests are welcome. Run the checks in the Quick Start before opening a pull request. The RTL, the golden
model and the generated tables must stay in sync: `pytest sim/golden` fails if they drift.

## References

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness.* 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Maxim Milakov, Natalia Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

## License

MIT. See [LICENSE](LICENSE).
