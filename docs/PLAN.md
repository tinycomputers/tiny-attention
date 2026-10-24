# TinyAttention — End-to-End Build Plan

> **A real, synthesizable FlashAttention engine in SystemVerilog, compiled to WebAssembly,
> running live in the browser.** You can watch the online softmax keep its running max and sum,
> watch SRAM tiles fill and empty, and see why FlashAttention moves so much less data than standard attention.
>
> This is the fourth project in the series, after [tiny-gpu](https://github.com/deaneeth/tiny-gpu),
> [tiny-tpu](https://github.com/deaneeth/tiny-tpu) and tiny-cpu. It takes over the toolchain tiny-cpu
> already proved:
> `rtl/*.sv → Verilator → C++ harness → Emscripten → web/public/tiny_attn.wasm → React island → SVG`.

The house rule stays the same: **nothing on screen is faked.** Every PE value, running max, rescale factor,
SRAM tile and HBM byte that the UI draws comes from a `dbg_*` port on the compiled RTL.

---

## 1. Goals and non-goals

### Goals
1. **A real attention engine.** It computes `O = softmax(Q·Kᵀ / √d) · V` in fixed-point hardware:
   two 4×4 systolic matmul engines, a row-max tree, `exp2` units, an online-softmax state
   (`m`, `l`) per row, an output accumulator that rescales, and a reciprocal unit for the final
   normalization.
2. **Two schedules on one chip.** `cfg_mode = NAIVE` runs standard attention, which writes the full
   N×N `S` and `P` matrices to off-chip memory. `cfg_mode = FLASH` runs FlashAttention-2, which tiles
   the work, uses online softmax and never lets `S` or `P` leave the chip. Both run on the same datapath, and only the
   controller schedule changes. The UI runs both side by side and counts every HBM byte.
3. **Bit-exact verification.** A Python fixed-point golden model reproduces every rounding step.
   The RTL must match it **bit for bit** at every tile boundary, in both modes, with and without the causal
   mask. A separate float reference (numpy) measures the quantization error and pins it.
4. **A visual explanation of attention.** The app has four levels: the softmax cell, the tile dataflow, the
   FlashAttention schedule, and naive vs flash. It also has presets whose attention heatmaps show recognizable
   patterns (previous-token head, attention sink, a toy sentence).
5. **Synthesizable.** It builds with Yosys + nextpnr for an ECP5 (ULX3S). The LUT, DSP and BRAM counts and fmax
   go in the README.

### Non-goals (v1)
- Single head only, head dimension `d = 4` (it matches the 4×4 array). Larger `d` through K-accumulation is
  a stretch goal.
- Forward pass only: no backward pass and no training.
- No floating-point datapath. Everything is INT8 and fixed point (see §2.3). FP8 or bf16 is a stretch goal.
- No KV-cache decode mode in v1. FlashDecoding (split-K) is a stretch goal.
- The engine is not wired to tiny-cpu or tiny-tpu in v1. That integration is `tiny-soc`, in §12.

---

## 2. Architecture

### 2.1 Sizes (all `parameter`s in `tiny_attn_pkg.sv`)

| Parameter | Value | Notes |
|---|---|---|
| `D` (head dim) | 4 | equals the array width |
| `BR`, `BC` (tile rows/cols) | 4, 4 | one Q block × one KV block = one 4×4 `S` tile |
| `NMAX` (max sequence length) | 32 | runtime `cfg_n` ∈ {4, 8, …, 32} (multiple of 4) |
| Tiles per row/col | `N/4` (≤ 8) | up to 64 tile pairs per run (36 when causal) |

### 2.2 Block diagram

```
            ┌──────────────────── off-chip ────────────────────┐
            │  HBM model: Q, K, V, O (+ S, P in NAIVE mode)     │
            │  1 word/cycle, cfg_hbm_latency cycles to first     │
            └───────────────▲───────────────────┬───────────────┘
                            │ DMA (byte counters)│
 ┌──────────────────────────┴───────────────────▼────────────────────────────┐
 │ on-chip SRAM:  Q tile │ K tile ×2 │ V tile ×2 │ S/Z tile │ P tile │ O tile │
 │                       (ping-pong when cfg_overlap_en)                     │
 └──┬─────────────┬────────────────────────────────────────────┬──────────────┘
    │ Q           │ K (as weights)                              │ V (as weights)
 ┌──▼─────────────▼──┐   S (int)   ┌──────────┐  Z (Q8.8)  ┌────▼─────────────┐
 │ MM1  4×4 systolic │────────────▶│ scale +  │───────────▶│ online softmax   │
 │  S = Q·Kᵀ         │             │ mask     │            │ rowmax → m_new   │
 └───────────────────┘             └──────────┘            │ α = 2^(m−m_new)  │
                                                           │ P = 2^(Z−m_new)  │
                                                           │ l = α·l + ΣP     │
                                                           └──┬────────┬──────┘
                                                        P (UQ0.8)   α │
                                                     ┌────────▼──┐  ┌──▼─────────────┐
                                                     │ MM2 4×4   │─▶│ O accumulator  │
                                                     │ P·V       │  │ O = α·O + P·V  │
                                                     └───────────┘  └──┬─────────────┘
                                                                       │ after last KV tile
                                                               ┌───────▼──────────┐
                                                               │ recip(l), O·1/l  │──▶ DMA → HBM
                                                               └──────────────────┘
```

### 2.3 Number formats (the core of the design; record every rounding rule in `docs/ARCHITECTURE.md`)

The softmax uses **base 2**: `softmax(x) = 2^(x·log₂e) / Σ…`. The `log₂e/√d` factor and the
quantization scales fold into one host-computed multiplier `cfg_scale`. Hardware only ever computes `2^x`.
FlashAttention-3 does the same.

| Signal | Format | How it is produced |
|---|---|---|
| `Q, K, V` | INT8 signed | host-quantized (symmetric) |
| `S = Q·Kᵀ` | INT20 signed | MM1 accumulators (`4 · 127 · 127` needs 18 bits) |
| `cfg_scale` | UQ8.16 | `sq · sk · log₂e / √d`, computed by the host |
| `Z` (log2-domain logit) | Q8.8 signed, saturating | `sat16((S · cfg_scale) >>> 16)`; masked → `−128.0` (`0x8000`) |
| `m` (running row max) | Q8.8 | init `−128.0`; `m_new = max(m, rowmax(Z))` |
| `exp2(x)`, `x ≤ 0` | in Q9.8, out UQ1.15 | `n = x >>> 8`, `f = x & 0xFF`; `LUT256[f] >> −n`; `0` if `−n ≥ 16` |
| `α` | UQ1.15 | `exp2(m − m_new)` |
| `P` | UQ0.8 | `min(255, (exp2(Z − m_new) + 64) >> 7)` |
| `l` (row sum) | UQ16.16 | `((l · α) >> 15) + (rowsum(P) << 8)` |
| `O` | Q15.16 signed (32-bit) | `((O · α) >>> 15) + (MM2(P, V) << 8)` |
| `1/l` | mantissa + exponent | normalize `l` (leading-zero count), LUT64 seed, one Newton-Raphson step |
| `out` | Q8.8 signed | `round(O · (1/l))`, saturated; host dequantizes with `sv` |

Rules:
- `>>>` is arithmetic shift (floor). Every place that rounds says so explicitly in `ARCHITECTURE.md`.
- `exp2` and `recip` ROM contents are **generated by Python** (`sim/golden/gen_luts.py` → `rtl/mem/*.mem`).
  RTL and golden model read the same file, so there is one source of truth.
- `l` sums the **quantized** `P` (the same values MM2 uses), so the numerator and denominator agree.
- **NAIVE ≠ FLASH bit for bit.** NAIVE takes the max over the full row and never rescales. FLASH rescales by α
  after each tile, which adds rounding. The golden model implements both exactly. The UI shows the
  difference, usually a few LSBs, and the docs explain why.

### 2.4 Schedules

**FLASH (FlashAttention-2 loop order: Q blocks outer, so `O`, `m`, `l` stay on chip):**
```
for qi in 0 .. N/4-1:
    DMA Q[qi] → SRAM;  m[:] = −128.0;  l[:] = 0;  O[:] = 0
    for kj in 0 .. N/4-1:
        if causal and kj > qi: break               # fully masked tile: never loaded
        DMA K[kj], V[kj] → SRAM                     # prefetch kj+1 if cfg_overlap_en
        S  = MM1(Q[qi], K[kj])
        Z  = mask(scale(S))                         # diagonal tile only is partially masked
        m_new = max(m, rowmax(Z));  α = exp2(m − m_new)
        P  = q8(exp2(Z − m_new))
        l  = α·l + ΣP;   O = α·O + MM2(P, V[kj]);   m = m_new
    DMA (O · recip(l)) → HBM
```

**NAIVE (standard attention; FlashAttention paper, Algorithm 0):**
```
pass 1: for every (qi, kj): S = MM1(Q[qi], K[kj]);  write Z tile → HBM     (N² × 2 B)
pass 2: for every row: read Z row → rowmax m                                (N² × 2 B)
pass 3: for every row: read Z row → P = q8(exp2(Z − m)), l = ΣP; write P   (N² × 2 B + N² × 1 B)
pass 4: for every (qi, kj): read P tile, V[kj]; O += MM2(P, V)             (N² × 1 B + …)
        O · recip(l) → HBM
```
Deferred normalization in both modes makes the comparison fair, because only memory traffic and rescaling differ.

### 2.5 Runtime configuration pins (one WASM build covers everything)

| Pin | Values | Effect |
|---|---|---|
| `cfg_n` | 4…32, step 4 | sequence length |
| `cfg_mode` | `NAIVE` / `FLASH` | controller schedule |
| `cfg_causal` | 0/1 | skips tiles above the diagonal and masks the diagonal tile |
| `cfg_overlap_en` | 0/1 | ping-pong K/V buffers; the DMA prefetches tile `kj+1` during compute on `kj` |
| `cfg_hbm_latency` | 1…64 | cycles to first word; makes the cost of memory visible |
| `cfg_scale` | UQ8.16 | see §2.3 |

### 2.6 Performance counters (exposed as `dbg_*`, pinned in tests)
`cycles`, `hbm_rd_bytes`, `hbm_wr_bytes`, `sram_rd`, `sram_wr`, `mm1_macs`, `mm2_macs`,
`exp2_ops`, `tiles_skipped`, and busy cycles per engine (MM1, softmax, MM2, DMA) for utilization bars.
The golden model also computes **analytical** HBM byte counts per mode, and the RTL counters must match them exactly.

---

## 3. Repository layout (mirrors tiny-cpu)

```
tiny-attention/
├── AGENTS.md / CLAUDE.md / CONTEXT.md
├── README.md  LICENSE  .editorconfig  .gitignore
├── .tools/  setup.sh  emsdk.sh          # copied from tiny-cpu (Verilator 5.036 + emsdk, project-local)
├── scripts/env.sh                         # copied from tiny-cpu
├── .github/workflows/
│   ├── rtl.yml        # lint + golden tests + unit + tile lockstep + e2e random + perf
│   ├── rtl-nightly.yml
│   ├── wasm.yml       # build, size budget, node test
│   ├── web.yml        # lint, typecheck, vitest, build, Playwright
│   └── fpga.yml       # yosys/nextpnr ECP5, posts LUT/DSP/BRAM/fmax (Phase 10)
├── rtl/
│   ├── tiny_attn_pkg.sv     # params, formats, enums, structs
│   ├── pe.sv                # INT8×INT8 MAC (+ signed/unsigned A operand param)
│   ├── systolic_array.sv    # 4×4 weight-stationary (adapted from tiny-tpu)
│   ├── mm_engine.sv         # array + skew/deskew + weight load FSM
│   ├── scale_mask.sv        # S → Z, causal mask
│   ├── row_max.sv           # 4-input max tree
│   ├── exp2_unit.sv         # LUT256 + shifter
│   ├── online_softmax.sv    # m, l, α, P per row lane
│   ├── o_accumulator.sv     # O = α·O + PV
│   ├── recip_unit.sv        # LZC + LUT64 + 1 Newton-Raphson step
│   ├── normalize.sv         # O · 1/l → Q8.8
│   ├── sram_bank.sv         # sync-read BRAM-style buffer
│   ├── hbm_model.sv         # word memory + latency pipeline + byte counters
│   ├── dma.sv               # tile moves HBM ↔ SRAM (row-major, strided)
│   ├── controller.sv        # FLASH and NAIVE schedules, causal skip, prefetch
│   ├── perf_counters.sv
│   ├── tiny_attn_top.sv     # cfg_* pins, host load port, dbg_* ports
│   ├── mem/                 # exp2_lut.mem, recip_lut.mem (generated, checked in)
│   └── spike/counter.sv     # toolchain smoke test, as in tiny-cpu
├── sim/
│   ├── Makefile  conftest.py  requirements.txt
│   ├── golden/
│   │   ├── fixed.py         # bit-exact model: exp2, recip, online softmax, both schedules
│   │   ├── float_ref.py     # numpy float attention (ground truth for error)
│   │   ├── gen_luts.py      # writes rtl/mem/*.mem
│   │   ├── traffic.py       # analytical HBM bytes per mode/N/causal
│   │   ├── presets.py       # preset Q/K/V generators (also exported to web)
│   │   └── test_golden.py   # golden self-tests (pytest, no RTL)
│   ├── test_pe.py  test_systolic_array.py  test_mm_engine.py
│   ├── test_exp2.py         # EXHAUSTIVE over every Q9.8 input ≤ 0
│   ├── test_recip.py        # EXHAUSTIVE over every normalized mantissa
│   ├── test_row_max.py  test_scale_mask.py  test_online_softmax.py  test_o_accumulator.py
│   ├── test_sram_dma.py  test_hbm_model.py
│   ├── test_tile_lockstep.py   # RTL vs golden at every tile boundary
│   ├── test_e2e.py             # full attention, random seeds × modes × causal × overlap
│   ├── test_accuracy.py        # fixed golden vs float: pinned error bounds
│   ├── test_perf.py            # exact cycles + HBM bytes vs perf_baseline.json / traffic.py
│   └── perf_baseline.json
├── wasm/
│   ├── harness.cpp  bindings.cpp  emscripten_stubs.cpp  build.sh  test.mjs  README.md
├── fpga/ecp5/                  # ULX3S top (UART for host load/readback), lpf, Makefile
├── docs/
│   ├── PLAN.md                 # this file
│   ├── ARCHITECTURE.md         # decisions, formats, rounding rules, timing diagrams
│   ├── STATE_SCHEMA.md         # per-cycle contract (§6)
│   ├── VERIFICATION.md         # layers, measured error, coverage
│   └── NUMERICS.md             # error analysis write-up (feeds the docs page)
└── web/                        # Astro + React + Tailwind + shadcn (copied from tiny-cpu)
    ├── public/tiny_attn.mjs  tiny_attn.wasm
    └── src/
        ├── lib/  state-schema.ts  wasm-loader.ts  presets.ts  quantize.ts  formats.ts
        ├── hooks/useTinyAttn.ts      # lifecycle, step/run/rewind, history ring
        ├── components/ (see §8)
        └── pages/ index.astro  app.astro  docs/*.astro
```

---

## 4. Toolchain

| Tool | Version | Used for |
|---|---|---|
| Verilator | 5.036 (project-local, `.tools/setup.sh`) | lint, cocotb, WASM |
| cocotb + pytest + numpy | `.venv` | all RTL and golden tests |
| Emscripten | `.tools/emsdk.sh` | WASM |
| Yosys + nextpnr-ecp5 + prjtrellis | OSS CAD Suite | Phase 10 |
| pnpm + Astro + React + Vitest + Playwright | as tiny-cpu | web |

There is no C compiler or assembler this time. The host side is TypeScript: `quantize.ts` turns float matrices into
INT8 values plus scales and computes `cfg_scale`. It is tested against the Python `quantize` in `fixed.py` with shared
JSON fixtures.

---

## 5. Verification strategy

Five layers. Each phase says which layers must be green.

1. **Unit tests (cocotb), one per module.** `exp2_unit` and `recip_unit` are tested **exhaustively**
   (all 65 537 Q9.8 inputs ≤ 0, and every normalized `l` mantissa) against `fixed.py`. The PE and array get random vectors against numpy. `row_max`,
   `scale_mask` (saturation edges, mask on and off), `online_softmax` (sequences of tiles including
   `m` rising, falling and staying equal), `o_accumulator` (α = 1, α → 0, sign edges) and DMA
   (every tile index, strides, latency 1 and 64) all have their own tests.
2. **Tile-boundary lockstep.** `fixed.py` emits a trace after each `(qi, kj)`: `S`, `Z`, `m`, `α`, `P`,
   `l`, `O`. The test reads the same values from `dbg_*` and stops at the **first divergence** with a readable diff,
   for example: `tile (qi=1,kj=2) row 3: α expected 0x5a82 got 0x5a81`.
3. **End-to-end random.** Random `Q/K/V` × `N ∈ {4…32}` × `mode` × `causal` × `overlap` × `hbm_latency`.
   The output must be bit-exact against `fixed.py`. CI runs 200 seeds; nightly runs 5000 and uploads the failing seed and its FST.
   Weighted cases: all-equal logits, one dominant logit, saturated `Z`, max-magnitude V, `N = 4` (a single tile).
4. **Accuracy against float.** `test_accuracy.py` checks fixed against `float_ref.py` over 1000 seeds. It pins max-abs
   and mean-abs error of `out` (in units of `sv`) and max `|P_fixed − P_float|`. The measured numbers go in
   `VERIFICATION.md` and in the README table. A second check pins the FLASH vs NAIVE difference.
5. **Performance.** Exact cycle counts per `(mode, N, causal, overlap, latency)` grid point are pinned in
   `perf_baseline.json`. HBM bytes must **equal** `traffic.py` exactly. An unexpected change fails CI.

---

## 6. State contract (`docs/STATE_SCHEMA.md` ↔ `web/src/lib/state-schema.ts`)

The harness emits one `CycleState` per clock. Every field documents its `dbg_*` source.

```ts
type CtrlState = "IDLE" | "LOAD_Q" | "LOAD_KV" | "MM1" | "SOFTMAX" | "MM2" | "RESCALE"
               | "NORMALIZE" | "STORE_O" | "N_WRITE_Z" | "N_ROWMAX" | "N_EXP" | "N_PV" | "DONE";

interface PEState { w: number; a: number; psum: number; active: boolean }

interface SramBuf {
  name: "Q" | "K0" | "K1" | "V0" | "V1" | "Z" | "P" | "O";
  tag: string | null;          // e.g. "K[2]"; derived from dbg_buf_tag (tile index)
  fill: number;                // words loaded, 0..16
  rd: boolean; wr: boolean;    // accessed this cycle
}

interface CycleState {
  cycle: number;
  cfg: { n: number; mode: "NAIVE" | "FLASH"; causal: boolean; overlap: boolean; hbmLatency: number };
  ctrl: { state: CtrlState; qi: number; kj: number; skipped: boolean };
  dma:  { active: boolean; dir: "RD" | "WR" | null; buf: string | null; hbmAddr: number };
  sram: SramBuf[];
  mm1:  PEState[];             // 16, row-major
  mm2:  PEState[];
  softmax: {                   // one entry per row lane (4)
    z: number[]; rowMax: number; mOld: number; mNew: number;
    alpha: number; p: number[]; l: number;
  }[];
  oacc: number[];              // 16 (4 rows × d)
  recip: number[] | null;      // 4, only in NORMALIZE
  out:   { row: number; values: number[] } | null;   // written to HBM this cycle
  counters: { hbmRd: number; hbmWr: number; sramRd: number; sramWr: number;
              mm1Macs: number; mm2Macs: number; exp2: number; tilesSkipped: number;
              busy: { mm1: number; softmax: number; mm2: number; dma: number } };
  done: boolean;
}
```

Raw fixed-point integers cross the boundary. `formats.ts` converts them to reals for display, and
that conversion is documented as **derived**.

---

## 7. WASM harness API

```cpp
class TinyAttnSim {
  void configure(val cfg);                 // {n, mode, causal, overlap, hbmLatency, scale}
  void loadQKV(val q, val k, val v);       // Int8Array each, row-major N×4, via host load port
  void reset();
  val  step();                             // one clock → CycleState
  val  run(int maxCycles);                 // CycleState[] until done or limit
  val  runFast(int maxCycles);             // summary + counters only
  val  readOutput();                       // Int16Array N×4 (Q8.8)
  val  readHbm(int addr, int len);         // inspect S/P matrices in NAIVE mode
  val  getCounters();  bool isDone();
};
```

- **Several instances per page.** L4 runs NAIVE and FLASH at the same time, so each `TinyAttnSim` owns its own
  `VerilatedContext` and model. This is checked in the Phase 0 spike: two instances of one model, run independently.
- **Rewind is replay**, as in tiny-cpu: a ring buffer of 4096 states in `useTinyAttn`. Seeking further back
  re-runs from reset with `runFast`.
- **Size budget:** under 500 KB `.wasm` gzipped, enforced by `wasm.yml`.
- `wasm/test.mjs` runs three fixtures (one per mode, plus causal) and checks the output against golden JSON
  exported by `fixed.py`, and cycles and bytes against `perf_baseline.json`. The WASM gets the same check as the RTL.

---

## 8. Web app (`web/`)

It copies tiny-cpu's Astro, React, Tailwind and shadcn setup and visual language, so the four projects look like one family.

### Pages
- `/`: landing page with a hero loop of the FLASH schedule sweeping the N×N tile grid, then "Open the engine".
- `/app`: the visualizer (one React island).
- `/docs/what-is-attention`, `/docs/online-softmax` (the derivation with a live single-row demo),
  `/docs/flashattention` (IO-awareness, the traffic math), `/docs/fixed-point` (the exp2 trick, LUTs,
  rounding, measured error), `/docs/architecture`, `/docs/verification`.

### Layout
```
┌──────────────────────────────────────────────────────────────────────────┐
│ [Preset ▾ toy sentence] N [16] [Mode: FLASH|NAIVE] [Causal ●] [Overlap ●] │
│ HBM latency [8]   ⏮ ◀ ▶ ⏭ ▶▶ speed                                         │
├─────────────────────────┬────────────────────────────────────────────────┤
│ Inputs: Q K V editors   │ LevelTabs: L1 Softmax | L2 Tile | L3 Schedule | │
│ (float → INT8 preview)  │            L4 Naive vs Flash                   │
│                         │ ┌────────────────────────────────────────────┐ │
├─────────────────────────┤ │               active view                  │ │
│ Attention heatmap P     │ └────────────────────────────────────────────┘ │
│ (with token labels)     │ HBM bytes │ SRAM buffers │ utilization │ error │
└─────────────────────────┴────────────────────────────────────────────────┘
```

### Levels
| Level | Component | What it shows |
|---|---|---|
| **L1 Softmax cell** | `SoftmaxCellView.tsx` | One row lane. Each KV tile adds 4 logits. The plot shows `m` stepping up, `α` scaling the old sum down, `l` growing, and the exp2 unit splitting `x` into integer shift + LUT fraction. A side panel compares against a two-pass softmax of the same row, in float. |
| **L2 Tile dataflow** | `TileView.tsx` | The datapath for one `(qi, kj)` step: MM1 PEs filling with skewed Q and K, `S` → `Z`, the row-max tree, 4 exp2 lanes, P going into MM2, and the O accumulator being rescaled by α. Wires carry live values. |
| **L3 FlashAttention schedule** | `ScheduleView.tsx` (**signature visual**) | The N×N attention matrix as a grid of 4×4 tiles. The current tile lights up, finished tiles fade, causal tiles above the diagonal are greyed out and skipped. Next to it, the SRAM buffers show their tags (`Q[1]`, `K[2]`, `V[2]`) filling and emptying, the DMA arrow points HBM ↔ SRAM, and a live HBM byte counter runs. With overlap on, you can see the ping-pong buffer loading `K[3]` while `K[2]` is computing. |
| **L4 Naive vs Flash** | `CompareView.tsx` | Two simulator instances on the same inputs. NAIVE fills a full `S` matrix in "HBM" while FLASH never does. Bars show HBM bytes, cycles, SRAM needed and utilization. An **N sweep** (4 → 32, using `runFast`) plots HBM bytes against N for both modes. An accuracy panel shows the FLASH vs NAIVE vs float error heatmaps. |

### Presets (`sim/golden/presets.py` → `web/src/lib/presets.ts`, generated)
Toy sentence with token labels (handcrafted Q/K so that a pronoun attends to its noun), previous-token head,
attention sink (everything attends to token 0), uniform, identity (self), random (seeded), and
custom (matrix editor adapted from tiny-tpu's `MatrixInput`).

### Other components
`Controls.tsx`, `LevelTabs.tsx` (reused), `MatrixEditor.tsx`, `AttentionHeatmap.tsx`,
`SramPanel.tsx`, `HbmMeter.tsx`, `UtilizationBars.tsx`, `ErrorPanel.tsx`, `FixedPointTooltip.tsx`
(hover any value: raw hex, format, real value).

### Web tests
- Vitest: `quantize.test.ts` (shared JSON fixtures with Python), `formats.test.ts`,
  `state-schema.test.ts` (a recorded `CycleState[]` fixture validated with zod), `presets.test.ts`.
- Playwright smoke test: load `/app`, run the toy sentence in FLASH, and assert the heatmap's argmax for the pronoun
  row is the noun column and that the HBM counter equals the golden number.

---

## 9. Phases, deliverables and exit criteria

The estimates assume one person part-time, at tiny-cpu's pace.

### Phase 0: Scaffold + toolchain spike (½ week)
- Create the repo with the §3 layout. Copy `.tools/`, `scripts/env.sh`, `wasm/emscripten_stubs.cpp`, the CI skeletons
  and the `web/` config from tiny-cpu, so its Phase 0 gotchas are already solved.
- Spike: `counter.sv` → WASM, with **two independent instances** of one model on one page.
- **Exit:** all workflows are green on an empty project; the spike page runs two counters at different speeds.

### Phase 1: Golden models + numerics (1.5 weeks), *the most important phase*
- `float_ref.py`; `fixed.py` with `exp2`, `recip`, online softmax, both schedules and tile traces;
  `gen_luts.py`; `traffic.py`; `presets.py`; host `quantize`.
- Error study: sweep LUT sizes and P width. Before any RTL exists, confirm that the §2.3 formats give error you can
  defend. Write up `NUMERICS.md`. **If the formats change, they change here, not later.**
- Lock the decisions into `ARCHITECTURE.md` (formats, rounding, NAIVE ≠ FLASH note).
- **Exit:** `pytest sim/golden -q` is green; FLASH and NAIVE in `fixed.py` both agree with `float_ref` within the
  pinned bounds over 1000 seeds; the LUT `.mem` files are generated and checked in.

### Phase 2: Arithmetic building blocks (1.5 weeks)
- `tiny_attn_pkg`, `pe`, `systolic_array`, `mm_engine`, `exp2_unit`, `recip_unit`, `row_max`,
  `scale_mask`, each with its unit test (§5 layer 1, exhaustive where stated).
- **Exit:** `verilator --lint-only -Wall` is clean; every unit test passes; `exp2` and `recip` are exhaustively bit-exact.

### Phase 3: One tile, end to end (1.5 weeks)
- `online_softmax`, `o_accumulator`, `normalize`, plus a tile-level top fed directly by the testbench
  (no memory system yet): one Q block streams through several KV blocks.
- **Exit:** `test_tile_lockstep.py` is green for N = 4…32 on a single Q block; the waveform shows MM1 → softmax → MM2.

### Phase 4: Memory system + FLASH controller (2 weeks)
- `sram_bank`, `hbm_model`, `dma`, `controller` (FLASH + causal), `perf_counters`, `tiny_attn_top` with
  the host load port and every `dbg_*` port.
- Write the controller timing diagram in `ARCHITECTURE.md` **before** coding it.
- **Exit:** full attention is bit-exact for 200 seeds × N × causal; HBM bytes equal `traffic.py`.

### Phase 5: NAIVE mode + overlap + perf pinning (1 week)
- NAIVE schedule (Z and P written to and read back from HBM), `cfg_overlap_en` ping-pong prefetch, and the `cfg_hbm_latency` sweep.
- **Exit:** the whole random matrix (§5 layer 3) is green; `perf_baseline.json` is pinned; the accuracy test is green.
  *At this point the engine is fully verified hardware and worth publishing on its own.*

### Phase 6: WASM harness (1 week)
- `harness.cpp`, `bindings.cpp`, `build.sh`, `STATE_SCHEMA.md`, `test.mjs` with golden fixtures.
- **Exit:** `wasm.yml` is green; the size budget is met; two instances run independently in Node.

### Phase 7: Core UI + L3 Schedule (2 weeks)
- `useTinyAttn`, `Controls`, `MatrixEditor`, presets, `quantize.ts`, `AttentionHeatmap`, `SramPanel`,
  `HbmMeter`, and **L3 Schedule first**.
- **Exit:** pick a preset, step forward and back, and watch tiles, buffers and bytes update; the Playwright smoke test passes.

### Phase 8: Remaining levels (1.5 weeks)
- L1 Softmax cell, L2 Tile dataflow, L4 Naive vs Flash (two instances, N sweep, error panel).
- **Exit:** every level works on desktop and at 375 px, in light and dark themes.

### Phase 9: Docs, landing page, launch (1 week)
- The six docs pages, each with an embedded live mini-visualizer. Hero capture and OG image (reuse the tiny-tpu scripts).
- README in the series style: live demo, verification table (bit-exact seeds, exhaustive unit coverage, measured
  error, HBM savings at N = 32), and FPGA numbers.
- Deploy to `tiny-attention.vercel.app`.
- **Exit:** the public URL is live; all CI is green.

### Phase 10: FPGA synthesis (½–1 week, can run alongside Phases 7–9)
- `fpga/ecp5` for the ULX3S: 32 INT8 MACs map onto DSP blocks, and the LUTs and SRAM go to BRAM. Host load and readback use UART.
- `fpga.yml` posts LUT/FF/DSP/BRAM and fmax to the job summary; L4 uses fmax for "real time" estimates.
- **Exit:** synthesis is clean with no latches; the numbers are in the README. With a board, the toy sentence runs over UART.

**Total: roughly 14–15 weeks part-time.** Phases 0–5 (about 8 weeks) give you a bit-exact, verified FlashAttention
engine.

---

## 10. CI matrix

| Workflow | Trigger | Jobs |
|---|---|---|
| `rtl.yml` | push/PR touching `rtl/ sim/` | lint → golden tests → LUT freshness (`gen_luts.py` diff is empty) → unit → tile lockstep → e2e (200 seeds) → accuracy → perf |
| `rtl-nightly.yml` | cron | e2e with 5000 seeds; uploads the failing seed + FST |
| `wasm.yml` | push touching `rtl/ wasm/` | build → size budget → `test.mjs` |
| `web.yml` | push touching `web/` | lint → typecheck → vitest → presets freshness → build → Playwright |
| `fpga.yml` | push touching `rtl/ fpga/` | yosys + nextpnr-ecp5 → job summary |

**Full pre-PR check** (goes in `CLAUDE.md` once Phase 2 lands):
```bash
verilator --lint-only -Wall rtl/*.sv
pytest sim/golden -q
cd sim && make unit && make tile && make e2e SEEDS=50 && make accuracy && make perf
bash wasm/build.sh && node wasm/test.mjs
cd web && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fixed-point formats turn out too coarse (visible error, P collapsing to 0) | Phase 1 error study before any RTL; formats live in one package and one Python module |
| Python and RTL disagree on rounding | One generated LUT source; every rounding rule written down; exhaustive unit tests on `exp2`/`recip`; tile lockstep catches drift at the first tile |
| Overflow in `l` or `O` at N = 32 with extreme inputs | Bit-width proofs in `ARCHITECTURE.md`; directed max-magnitude tests; `dbg_overflow` sticky flag asserted low in every test |
| Controller off-by-one with sync SRAM + HBM latency | Timing diagram first; latency is swept 1…64 in random tests |
| Large `CycleState` (two arrays + softmax lanes) slows the UI | `runFast` for unrendered cycles, 4096-entry ring buffer, typed arrays inside the harness |
| FLASH vs NAIVE mismatch reported as a "bug" | Documented and pinned as an expected difference with its own bound and docs section |
| Scope creep (multi-head, decode, FP8) | All of it is in §12; nothing from it starts before Phase 9 ships |

---

## 12. Stretch goals (after v1 ships)

1. **Larger head dim** (`d = 8, 16`) by accumulating MM1 over d-chunks (tiny-tpu's L3 tiling, reused).
2. **Engine-level pipelining**: MM1 on tile `kj+1` overlaps softmax/MM2 on tile `kj`, as in FA-3 warp
   specialization. Shown as a Gantt view of engine busy time.
3. **KV-cache decode + FlashDecoding (split-K)**: one query row, KV split across parallel lanes, then a
   log-sum-exp merge. Adds a "decode" mode to L3.
4. **Multi-head / GQA**: heads are time-multiplexed; K/V are shared across query groups to show the bandwidth savings.
5. **Masks and positions**: sliding window, ALiBi bias unit, RoPE rotation unit in front of MM1.
6. **FP8 (E4M3) inputs** with a comparison of error against INT8.
7. **tiny-soc**: tiny-cpu issues attention jobs over MMIO; tiny-tpu does the projections (`X·Wq/Wk/Wv`);
   tiny-attention does the attention. That is one transformer block in hardware.
8. **Tiny Tapeout** submission of the softmax lane (exp2 + online state).

---

## 13. First week, concretely

1. Create `deaneeth/tiny-attention`. Copy `.tools/`, `scripts/env.sh`, `.editorconfig`, the workflow skeletons, `web/` config
   and `wasm/emscripten_stubs.cpp` from tiny-cpu. Write `CLAUDE.md`, `AGENTS.md` and `CONTEXT.md`.
2. Spike: two independent `counter.sv` model instances in one WASM page; get CI green.
3. `float_ref.py` + `fixed.py` exp2 and recip with `gen_luts.py`; the first numerics plot (exp2 error vs LUT size).
4. Online softmax in `fixed.py` for a single row, checked against float. Then the full FLASH and NAIVE schedules.
5. Draft `ARCHITECTURE.md`: §2.3 formats with bit-width proofs, plus the MM1 → softmax → MM2 tile timing diagram.
