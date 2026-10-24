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

## 8. Open until Phase 4

- Controller timing diagram for synchronous SRAM + HBM latency (written before the controller RTL).
- SRAM sizing for `cfg_qrows = 32`: `m, l` 32 entries each, `O` 32 × 4 × 26 bit ≈ 400 B, plus 32 Q rows.
