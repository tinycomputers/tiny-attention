# TinyAttention: Architecture decisions

Source of truth for number formats and rounding. `sim/golden/fixed.py` implements every rule below
and the RTL must match it bit for bit. Values here come from the Phase 1 numerics study
(`docs/NUMERICS.md`, raw output in `sim/golden/study_results_1000seeds.txt`); the study, not intuition, picked them. Change a rule here first, then in `fixed.py`,
then in the RTL.

Status: formats and rounding **frozen at the end of Phase 1**. The controller timing diagram is added
at the start of Phase 4 (see the end of this file).

## 1. Decisions table

| Decision | Choice | Why (measured in NUMERICS.md) |
|---|---|---|
| Head dim, tile | `D = 4`, 4×4 tiles, `NMAX = 32` | matches the 4×4 array |
| Softmax base | 2. `log₂e/√D` and both quant scales fold into one host constant `cfg_scale` | hardware only needs `2^x` |
| Q, K, V | INT8, symmetric, clipped to ±127 | |
| `P` (probabilities) | **UQ1.8**: 0..256, 9 bits unsigned (`256` = 1.0) | the dominant error source; 8→7 bits nearly doubles the mean error (0.069→0.125 V-LSB) and raises the worst case from 1.3 to 2.1 |
| Rounding | **truncate** (floor) everywhere except the two `P`/`out` roundings listed in §3 | rounding every step vs truncating every step changes the mean error by 2% (0.0677 vs 0.0691) |
| `l`, `O` extra fraction bits | `LF = OF = 4` | 4 bits loses nothing vs 8 (0.0691 vs 0.0683); 2 bits costs +5% mean error |
| `cfg_scale` | 24-bit **UQ4.20** | UQ8.16 gave 1.14 V-LSB worst error on small-magnitude inputs, UQ4.20 gives 0.08 |
| Reciprocal | LUT64 seed + **1** Newton-Raphson step | 0 steps: mean error +83%; 2 steps: no gain |
| FLASH vs NAIVE | not bit-identical; FLASH is the **more** accurate one | see §5 |
| On-chip Q block | `cfg_qrows` ∈ {4, 8, 16, 32} rows resident per Q block | traffic falls with it (§6); output does not depend on it |

## 2. Number formats and bit widths

`>>` is an arithmetic (floor) shift. Every width below is proven by the bound in the last column and
re-checked by `test_bit_width_bounds_worst_case`.

| Signal | Format | Width | Bound |
|---|---|---|---|
| `Q, K, V` | int8 | 8 | clipped to ±127 |
| `S = Q·Kᵀ` | signed int | 17 | `|S| ≤ 4·127·127 = 64 516 < 2^16` |
| `cfg_scale` | UQ4.20 | 24 | host-computed; `quantize()` raises if it would not fit |
| `S·cfg_scale` | signed | 41 | `< 2^16 · 2^24 = 2^40` |
| `Z` | Q8.8 signed, saturating to ±32767 | 16 | `0x8000` is reserved for **masked** |
| `m` | Q8.8 | 16 | init `0x8000` (−128.0) |
| `x = Z − m` | Q9.8, always ≤ 0 | 17 | `−65535 ≤ x ≤ 0` |
| `exp2(x)`, `α` | UQ1.15 | 16 | `≤ 32768` |
| `exp2` LUT | 256 × 16-bit, `round(2^15 · 2^(f/256))` | 16 | 32768..65358 |
| `P` | UQ1.8 | 9 | `≤ 256` |
| `l` | integer = real `l` × 2^(8+LF) | 18 unsigned | `l ≤ 32 · 256 · 2^4 = 2^17`; observed max = 131072 |
| `l·α` | | 34 | |
| `MM2 = P·V` (one tile) | signed | 20 | `4 · 256 · 127 = 130 048` |
| `O` | integer = ΣP·V × 2^OF | 26 signed | `32 · 256 · 127 · 2^4 = 16 646 144 < 2^25`; observed max = 16 646 144 |
| `O·α` | | 42 | |
| `recip(l)` | `r` UQ1.16 + exponent `k` | 17 + 5 | `r ∈ (2^15, 2^16]` |
| `O·r` | signed | 43 | |
| `out` | Q8.8 signed, units of V-LSB | 16 | `|out| ≤ 127 · 256 = 32 512` |

## 3. Datapath equations (what `fixed.py` does, in order)

1. **Scale and mask.** `Z = sat(( S · cfg_scale ) >> 12)` clipped to ±32767. (`>> (SF − 8)` with `SF = 20`.)
   Masked entries (causal, `col > row`) are forced to `0x8000`. A masked entry never wins the max and its `P` is forced to 0.
2. **Online max.** `m_new = max(m, rowmax(Z))`.
3. **α.** `α = exp2(m − m_new)`.
4. **exp2.** For `x ≤ 0` in Q9.8: `s = −(x >> 8)`, `f = x & 0xFF`. Result `= LUT[f] >> s`, truncating,
   and exactly 0 when `s ≥ 16`.
5. **P.** `P = (exp2(Z − m_new) + 64) >> 7`. **Rounds to nearest** (UQ1.15 → UQ1.8). `exp2 = 32768` gives `P = 256`.
6. **Sum.** `l = ((l · α) >> 15) + (ΣP << LF)`, truncating.
7. **PV.** `O = ((O · α) >> 15) + (MM2(P, V) << OF)`, truncating.
8. **Normalize (after the last K/V tile).** Let `k = bitlen(l) − 1`.
   Mantissa `M = (l << 15) >> k` (or `l >> (k−15)` when `k > 15`) as UQ1.15.
   Seed `r0 = RECIP_LUT[(M >> 9) & 63]` (UQ0.16). One Newton step:
   `t = (M · r0) >> 15; r = (r0 · (2^17 − t)) >> 16`.
   `out = (O · r + 2^(7+k)) >> (8 + k)`, **rounds to nearest**, clipped to int16.
9. The host dequantizes: `real = out / 256 · sv`.

Only two operations round to nearest (steps 5 and 8). Everything else truncates. Both are cheap (one adder)
and measurably matter: step 5 rounds the dominant error source, step 8 is the final result.

## 4. Schedules

FLASH and NAIVE share the same arithmetic units; only the controller differs. Both use **deferred
normalization**: `l` and `O` are divided only at the very end, so the comparison isolates memory traffic
and rescaling.

- **FLASH (FlashAttention-2 order).** Q blocks outer. For each block of `cfg_qrows` rows: load its Q tiles,
  then for each K/V tile load it **once** and apply it to every row group in the block. `m, l, O` stay on chip,
  `S` and `P` never leave the chip. Causal: columns above the block's last row are never loaded; inside the block
  a tile above a row group's diagonal is skipped.
- **NAIVE (standard attention, 3-pass safe softmax).** Pass 1 writes `Z` tiles to HBM. Pass 2 reads them back
  for the row max. Pass 3 reads them again, computes `P` (also needs row sums), writes `P` to HBM.
  Pass 4 reads `P` and `V`, computes `O`, normalizes. Row statistics `m` and `l` (N entries each) live on chip.
  HBM element sizes: `Z` 2 B, `P` 2 B (9 bits), `O` 2 B, `Q/K/V` 1 B.

## 5. FLASH vs NAIVE accuracy

Not bit-identical, and the plan's prediction was backwards. FLASH quantizes each tile's `P` against the
**running** max, which is usually smaller than the final max, so early tiles keep more significant bits and are
rescaled by `α` afterwards. NAIVE quantizes every `P` against the **global** max, so small probabilities lose
resolution. Measured over 1000 seeds (peaky inputs, N=32): FLASH max error 1.26 V-LSB, mean 0.069; NAIVE max error 1.74, mean 0.099.
The two outputs differ by at most 1.3 V-LSB over 1000 seeds. Pinned in `test_flash_vs_naive_difference_bounded`.

## 6. HBM traffic

Closed form in `sim/golden/traffic.py`, checked against the byte counters of both schedules. `T` = tile pairs
computed (`nb²`, or `nb(nb+1)/2` causal).

| Mode | Bytes |
|---|---|
| FLASH | `Q: 16·nb`; `K+V: 32 · kv_tile_loads(qrows)`; `O: 32·nb` |
| NAIVE | reads `16·nb + 16T + 32T + 32T + 48T`; writes `32T + 32T + 32·nb` |

N = 32, non-causal:

| | bytes | vs NAIVE (12 672 B) |
|---|---|---|
| FLASH, `qrows = 4` | 2 432 | 5.2× less |
| FLASH, `qrows = 8` | 1 408 | 9.0× |
| FLASH, `qrows = 16` | 896 | 14.1× |
| FLASH, `qrows = 32` | 640 | 19.8× (K and V read once each: the minimum) |

**Why this matters for the UI.** With only 4 rows on chip, both schedules reload K/V per Q block and the ratio
levels off near 6×. FlashAttention's real asymptotic win comes from a large on-chip Q block, which is why
`cfg_qrows` is a runtime knob and L4 shows it.

## 7. Phase 2 RTL contracts

Combinational arithmetic units (`exp2_unit`, `recip_unit`, `scale_mask`, `row_max`) are bit-exact with `fixed.py`;
the controller adds registers around them in Phase 3/4 as timing requires.

| Module | Contract |
|---|---|
| `pe` | `psum_out <= psum_in + a_in * w`, `a_out <= a_in`; one register each. `A_SIGNED=0` zero-extends (MM2's 9-bit P). |
| `systolic_array` | N×N weight-stationary. Weights load one row per cycle (`w_we_row[r]`); activations must arrive skewed. Column c holds C[i][c] during cycle `t_i + N + c`. |
| `mm_engine` | Skews A in, deskews C out. **Latency = 2N−1 = 7 cycles** from `a_valid` to `c_valid`, gaps allowed, invalid rows feed zeros. Weights must not change while rows are in flight. MM1: `A_W=8, A_SIGNED=1, ACC_W=17` (A = Q, W = Kᵀ). MM2: `A_W=9, A_SIGNED=0, ACC_W=18` (A = P, W = V). |
| `scale_mask` | `Z = sat((S·cfg_scale) >>> 12)` clipped to ±32767, `0x8000` when masked. |
| `row_max` | signed max of N values; `0x8000` never wins unless all inputs are masked. |
| `exp2_unit` | exact for every x ≤ 0 (verified over all 65 537 inputs); x > 0 saturates to `0xFFFF`. |
| `recip_unit` | exact for every UQ1.15 mantissa and every exponent; `l = 0` returns `r = 0, k = 0`. Two chained multiplies: register it before Phase 10 timing closure. |

Verification of this phase: `bash sim/lint.sh` (`-Wall`, every top, MM2 parameters) and `bash sim/run_unit.sh`
(PE, array, engine in both MM1 and MM2 configurations, row_max, scale_mask 30 000 random + edge cases,
exp2 exhaustive, recip exhaustive over every mantissa). A deliberate bug in the reciprocal Newton step or in the
engine's deskew delay makes the suites fail.

## 8. Phase 3: the tile core (`attn_tile`)

One 4-row Q block against a sequence of 4×4 K/V tiles, with no memory system yet. The testbench loads Q, Kᵀ and V
rows through ports; Phase 4's controller and DMA replace the testbench.

| Module | Contract |
|---|---|
| `online_softmax` | one row, combinational: `m_new, α, P[4], l_new` from `z[4], m_old, l_old`. Equals `fixed.online_softmax_step`. |
| `o_accumulator` | one row, 4 columns: `((O·α) >>> 15) + (PV <<< 4)`. Equals `fixed.o_update`. |
| `normalize` | one element: `(o·r + 2^(7+k)) >>> (8+k)`, clipped to int16. Equals `fixed.normalize`. |
| `attn_tile` | MM1 (A = Q rows, W = Kᵀ) → 4 `scale_mask` lanes → `online_softmax` → MM2 (A = P rows, W = V) → `o_accumulator`; after `tile_last`, one shared `recip_unit` and 4 `normalize` lanes emit the 4 output rows. |

**Tile schedule** (pinned in `test_attn_tile.py`; c0 = first cycle after `tile_start`):

| Cycle | Event |
|---|---|
| c0 … c0+3 | Q rows enter MM1 |
| c0+7 … c0+10 | S rows leave MM1; softmax stage runs on each, `m, l, α` registered |
| c0+8 … c0+11 | P rows enter MM2 (registered one cycle after the softmax stage) |
| c0+15 … c0+18 | PV rows leave MM2; `O` row updated |
| c0+19 | `tile_done` (the `dbg_*` tile values are valid) |

`tile_start` to `tile_done` is **20 cycles**, plus 4 cycles to load the weights, with no overlap between tiles.
Overlapping weight load and the next tile's MM1 with the current tile's MM2 is the Phase 5 `cfg_overlap_en`
feature; the 4-cycle weight load is the other thing it hides.

**Weight layout.** MM1 weights are Kᵀ, i.e. row r of Kᵀ is column r of the K tile; the SRAM read in Phase 4 is a strided
read. MM2 weights are the V tile rows as they are.

**Diagonal tile.** `tile_diag` makes lane j masked for softmax row i when `j > i`, which is the causal mask inside the
tile on the diagonal. Tiles above the diagonal are skipped by the controller and never reach this module.

**Debug ports** (each is a real register or wire, and they are what the visualizer's `CycleState` will carry):
`dbg_s_tile, dbg_z_tile, dbg_p_tile, dbg_alpha, dbg_m, dbg_l, dbg_o, dbg_srow`, and the weight, activation and partial-sum
state of every PE in both arrays (`dbg_mm1_*`, `dbg_mm2_*`).

**Verification.** `test_attn_tile.py` compares S, Z, P, α, m, l and O at every tile boundary and the normalized rows against
`fixed.flash(trace=True)`: 36 random blocks (N ∈ {4,8,16,32}, causal on and off, random `qi`, four input amplitudes),
corner cases (all logits equal with V = ±127, max |S|, saturating `cfg_scale`, near-zero `cfg_scale`, one-hot
attention, first causal row), and five Q blocks back to back to prove `blk_clear`. A wrong causal-mask boundary or
feeding the wrong row's α into the O update makes it fail with a diff naming the tile and the value.

## 9. Phase 4: memory system and FLASH controller (as built)

Verified by `sim/test_e2e.py`: bit-exact O, HBM bytes, tile count and K/V loads equal `traffic.py`, and busy
cycles equal `sim/golden/schedule.py`, for every N in {4..32}, causal on/off, every `cfg_qrows` in {4,8,16,32} with
`cfg_qrows ≤ N`, `cfg_hbm_latency` 0..64, and 200 random seeds (`SEEDS=200 bash sim/run_e2e.sh`).

### 9.1 Memory

| Block | What it is | Timing |
|---|---|---|
| `hbm_model` | 256 × 32-bit words; combinational reads; DMA write port; host load port; byte counters (4 B per DMA word, cleared by `start`) | no latency of its own |
| `dma` | one burst at a time; read or posted write | read: `cfg_latency` cycles then 1 word/cycle; write: 1 word/cycle, no latency |
| `sram_bank` | output staging, 32 rows × 64 bit, one write and one read port | **combinational read** (distributed-RAM style) |
| Q buffer (in `attn_tile`) | 32 rows × 32 bit | registers |
| K, V (in `tiny_attn_top`) | 4 words each | registers |

Read bursts of W words complete (done pulse visible) in cycle `L + W + 2` counting the start cycle as 0; write bursts
in `W + 1`. Pinned in `sim/test_dma.py`.

HBM layout (word addresses): Q rows 0..31 (one word = four int8), K rows 32..63, V rows 64..95, O rows 96..159
(two words per row: lanes 0-1, lanes 2-3, int16 each).

Deviation from the first plan: the SRAM is combinational-read here. A block-RAM mapping needs one prefetch stage on
the read side; that belongs to the FPGA task (Phase 9). Phase 4 counts the same cycles either way because no state
machine stage depends on the read latency.

### 9.2 Controller (`flash_ctrl`)

States: `IDLE → BLK → (LDQ, LDQ_W) → per K/V tile { LDK, LDK_W, LDV, LDV_W, WLOAD, per group TILE / TILE_W, TILE at end } → NORM, NORM_W → STO, STO_W → (next block | DONE)`.

- A block is `cfg_qrows / 4` row groups; the last block may be smaller.
- The K/V tile loop runs `kj = 0 .. last`, where `last = block's last group` when causal, else `N/4 − 1`.
- A K/V tile is DMA'd **once per block** and used by every unmasked group of the block. This is why `cfg_qrows`
  reduces traffic (see `traffic.py`), and why the output does not depend on it.
- Group tiles are issued one at a time. Each takes `tile_start` plus 20 cycles (`TILE` + `TILE_W` = 21 busy cycles).
  A masked (causal, `kj > group`) group costs one cycle in `TILE` and is not computed.

### 9.3 Cycle model (exact)

Busy cycles from the first state after `start` through `DONE`, inclusive (`sim/golden/schedule.py`):

```
1                                         DONE
per block (bg groups, gb first group):
  1 + 1 + (L + 4·bg + 2)                  BLK, Q load (4·bg words)
  per K/V tile kj (kj = 0 .. last):
    2 · (1 + L + 6)                       K load and V load (4 words each)
    4                                     weight load
    per group: 21 if tiled, 1 if masked
    1                                     end of K/V tile
  1 + (4·bg + 1)                          NORM, NORM_W
  1 + (8·bg + 1)                          STO, STO_W (posted, 8·bg words)
```

Example values from the model: N = 16, non-causal, `cfg_qrows = 4`: 737 cycles at L = 0, 1097 at L = 10.
N = 16, causal, `cfg_qrows = 16`, L = 10: 455 cycles (the causal skip is visible here).

The first plan's estimate of `2L + 32` per tile was wrong: it counted the DMA as `L + 4` and left out the state
transitions. The schedule above is what the RTL does.

### 9.4 Traffic (exact)

`traffic.py` FLASH: reads = `16·nb` (Q) + `32·kv_tile_loads` (K and V); writes = `32·nb` (O). Both are checked against
the byte counters of the running engine for every test run.

N = 32, non-causal (`traffic.py`):

| `cfg_qrows` | HBM bytes | vs NAIVE (12 672 B) |
|---|---|---|
| 4 | 2 432 | 5.2× |
| 8 | 1 408 | 9.0× |
| 16 | 896 | 14.1× |
| 32 | 640 | 19.8× |

### 9.5 Not in Phase 4

- `cfg_overlap_en` (DMA of the next K/V tile during compute) is Phase 5. The schedule above has no overlap, so the
  numbers in 9.3 are the Phase 4 baseline that Phase 5 will be measured against.
- NAIVE mode is Phase 5.
- The HBM model has no bandwidth limit beyond one word per cycle, and writes are not latency-modelled.

## 10. Phase 5: overlap, NAIVE schedule, perf baselines (design, written before RTL)

### 10.1 Overlap (`cfg_overlap_en`, FLASH only)

- K and V are double-buffered in `tiny_attn_top` (`kbuf[2]`, `vbuf[2]`, 4 words each). `cur` selects the buffer in use.
- A prefetch sequencer, separate from the main FSM, starts when the group loop of K/V tile `kj` begins and `kj` is not the
  block's last tile. It loads K then V of tile `kj+1` into buffer `!cur` over the one DMA engine.
- The main FSM, at the end of tile `kj`'s group loop, waits for the prefetch to finish (`C_WAITPF`, zero cycles if it has),
  swaps `cur`, then runs the weight load (4 cycles, serial: PE weights are stationary) and the next group loop.
- Not overlapped: the Q load, the first K/V tile of each block, and the O store.
- Only one DMA burst is in flight at a time, so the prefetch and the main FSM never issue DMA together. The protocol
  guarantees the prefetch is done before any block-boundary DMA.

### 10.2 NAIVE schedule (`cfg_mode = NAIVE`)

Same arithmetic as FLASH, with the normalization deferred as in the golden model (`fixed.naive`). Blocks are always
4 rows. Off-chip layout: Z tile `(qi,kj)` at `Z_BASE + 64·qi + 8·kj` (8 words, 2 int16 per word); P tile at `P_BASE + …`.
HBM grows to 2048 words (`HBM_AW = 11`).

| Pass | Per tile pair `(qi, kj)` | Block-level |
|---|---|---|
| 1 | load K tile, weight load, `OP_NZ`: MM1 + scale/mask → Z rows out; DMA write Z tile | Q block loaded once |
| 2 | DMA read Z tile → `OP_NM`: row max into `m` | `m` is final after the last kj |
| 3 | DMA read Z tile → `OP_NP`: `P = exp2(Z − m)`, `l += ΣP`; DMA write P tile | |
| 4 | DMA read P tile, load V tile, weight load, `OP_NPV`: MM2 with P, O += P·V with α = 1 | normalize, store O |

Masking, the `MASK` sentinel, and the causal diagonal are the same as the golden model. Pass 2 and pass 3 visit the
same tiles, so pass 2 finishes a block before pass 3 starts on it.

Traffic follows `traffic.py` NAIVE term by term: pass 1 reads `Q` once per block and `K` per tile; passes 2 and 3 each
read `Z` per tile; pass 3 writes `P` per tile; pass 4 reads `P` and `V` per tile; O is written once per block.

NAIVE has no overlap in Phase 5.

### 10.3 Tile-core operations

`tile_start` carries an operation:

| Op | Input | Output | State effect |
|---|---|---|---|
| `OP_FLASH` | Q (streamed), K/V weights | O rows (normalized at block end) | m, l, α, O (online) |
| `OP_NZ` | Q (streamed), K weights | Z rows on the output port | none |
| `OP_NM` | Z tile (`tin`) | none | m = max(m, rowmax) |
| `OP_NP` | Z tile (`tin`) | P rows on the output port | l += ΣP (α = 1 by construction) |
| `OP_NPV` | P tile (`tin`), V weights | none | O += P·V (α forced to 1) |

The output port (`out_*`) carries normalized O, Z or P rows; they go to the staging buffer, and the DMA writes them out.

### 10.4 Performance baselines

Phase 5 pins measured busy cycles for a grid of `(N, causal, qrows, L, overlap, mode)` in `sim/perf_baseline.json`.
The FLASH non-overlapped case stays exact against the closed-form model from Phase 4. The overlapped and NAIVE cases
are regression baselines measured from the RTL. Each one is checked by `sim/test_perf.py`; a change fails the test and
must be justified in review.

### 10.5 Measured results (as built)

From `sim/perf_baseline.json` (N = 32, non-causal, L = 10):

| Configuration | Busy cycles | vs sequential FLASH |
|---|---|---|
| FLASH, `cfg_qrows = 4`, no overlap | 4113 | 1.00 |
| FLASH, `cfg_qrows = 4`, overlap | 2937 | 0.71 |
| FLASH, `cfg_qrows = 16`, no overlap | 2133 | 0.52 |
| FLASH, `cfg_qrows = 16`, overlap | 1657 | 0.40 |
| NAIVE | 10961 | 2.66 |

Overlap saves about 29% at `cfg_qrows = 4` (the K/V loads are most of the time there) and about 22% at `cfg_qrows = 16`.
Causal runs follow the same pattern (2433 → 1845 at `cfg_qrows = 4`). NAIVE is 2.7× slower than FLASH at this size. Phase 5 does not break that gap down by pass; the pass
breakdown is a candidate for Phase 9 profiling.

The FLASH sequential case equals the cycle model in `sim/golden/schedule.py` exactly. The overlap and NAIVE cases are
pinned baselines; `test_e2e.py::perf_grid` also checks that overlap is never slower than sequential for the same
configuration.

Deviations from the first plan, recorded here so they are not forgotten:
- HBM grew from 256 to 2048 words to hold the NAIVE Z and P tiles.
- NAIVE runs block by block, not pass by pass over all blocks. Values and traffic totals are identical (the
  per-row computations are independent), but the sequencing differs from `fixed.naive`'s loop order.
- Overlap applies to FLASH only. NAIVE is sequential.

## 11. Phase 6: WebAssembly harness

- `wasm/harness/harness.{h,cpp}`: `TinyAttnSim`, one instance of `tiny_attn_top` with its own `VerilatedContext`. Drives
  the engine as a host would: load Q/K/V through the HBM load port, one-cycle start, read O through the readback port.
- `wasm/harness/bindings.cpp`: embind bindings. The JavaScript API and the per-clock snapshot are in `docs/STATE_SCHEMA.md`.
- `wasm/build.sh`: Verilator `--prefix Vtiny_attn_top`, `-O3`, then `em++`. `-Wno-UNOPTFLAT` is needed for the
  combinational paths through the staging buffer and the DMA word mux (no logic loop: the loop is through registers).
- Size: 206 KB raw, 81 KB gzipped. The budget is 500 KB gzipped, enforced by `wasm.yml`.

Results (`node wasm/test.mjs`, three golden fixtures interleaved on separate instances): O, HBM bytes, tiles, K/V loads and
busy cycles equal the golden values for FLASH sequential, FLASH with overlap and causal masking, and NAIVE with causal
masking. A fresh instance reproduces the run, so no state leaks between instances.

Deviation from the plan: the per-clock state exported to JavaScript is a subset (controller state, counters, byte counts).
Per-PE values, softmax lanes and SRAM tiles need wide-signal access and are the first item of Phase 7.

## 12. Phase 7: web visualizer (as built)

`/app` (`web/src/pages/app.astro`) runs the WASM engine in the browser and shows it clock by clock.

- **Presets** (`sim/golden/presets.py` → `web/src/lib/presets.ts`, generated): six attention patterns with token labels.
  Their Q/K/V are quantized in the browser by `web/src/lib/quantize.ts`, a mirror of `fixed.quantize`. Both are checked
  against golden values in `pytest` (freshness) and `vitest` (values).
- **Controls**: schedule (FLASH / NAIVE), causal, overlap (FLASH), `cfg_qrows`, HBM latency. Start, Step, Back (rebuilds
  the run from reset and advances N−1 clocks), Run to end, Reset.
- **Views**: the tile schedule (FLASH: row groups × K/V tiles, with the causal mask; NAIVE: pass-1 Z tiles), the current
  tile's S, Z, P, m, l, α and O (FLASH; NAIVE shows m, l and O only, since its S/Z/P never enter the tile core), HBM bytes,
  tile count, K/V loads, and controller state names (`FLASH_STATES` / `NAIVE_STATES`, matching the enums in the RTL).
- **Result**: the dequantized O, its maximum deviation from float attention (`web/src/lib/reference.ts`), and for NAIVE the
  probability matrix read back from HBM, with token labels.

Checks: `node wasm/test.mjs` (engine vs golden), `pnpm test` (quantization and tile order against the fixtures),
`pnpm typecheck`, `pnpm lint`, `pnpm build`. The browser run (built site, `astro preview`) showed no console errors; the
FLASH run and the NAIVE run both finished with max deviation 0.0049 on the toy preset; the phone width has no horizontal scroll.

Phase 7 items, all built: `MatrixEditor` (the Q, K and V values are editable; editing resets the run), `SramPanel` (the
K and V tile buffers, read from the RTL through `dbg_kbuf`, `dbg_vbuf`, `dbg_kv_sel`, with the buffer in use outlined),
`HbmMeter` (bytes so far against the expected total, from `web/src/lib/traffic-table.ts`, generated from
`sim/golden/traffic.py`), and `AttentionHeatmap` (engine probabilities from HBM for NAIVE; the float reference for FLASH,
which never stores P). The Playwright smoke test (`pnpm test:e2e`, five tests: FLASH output against float, step and back,
NAIVE probabilities, editing and re-running, phone width) runs against the production build.

Checks: `node wasm/test.mjs` (engine vs golden, now including the K/V buffer in use against the last tile loaded),
`pnpm test` (quantization, tile order), `pnpm test:e2e`, `pnpm typecheck`, `pnpm lint`, `pnpm build`, `pytest sim/golden`
(fixture, web-data and traffic-table freshness). The `astro dev` server hangs without output in this environment, so
browser checks use `astro build` + `astro preview`.
