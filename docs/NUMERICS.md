# TinyAttention: Numerics study (Phase 1)

Why the number formats in `docs/ARCHITECTURE.md` are what they are. Reproduce with
`python sim/golden/study.py 1000` (about 25 minutes; raw output in `sim/golden/study_results_1000seeds.txt`).

## Method

- Inputs: Gaussian Q, K (std `amp`) and Gaussian V (std 1), N × 4, fixed seeds. `soft` amp 1.5, `peaky` 2.5,
  `sharp` 3.5, `small` 0.3 (tiny magnitudes, which stress the scale constant). 1000 seeds per cell.
- All errors are in **V-LSB**: 1.0 = one INT8 step of V. For scale: rounding a perfect result to INT8 already costs
  up to 0.5.
- **`dp` (datapath error)** = fixed-point output vs float64 attention computed on the *dequantized INT8 inputs*.
  This is what the hardware is responsible for, and every choice below optimizes it.
- **`tot` (total error)** = vs float64 attention on the original float inputs. It is dominated by INT8
  quantization of Q and K (a 1% error on a logit of 30 moves it by 0.3), which no datapath choice can fix.
  It is reported so the README can be honest about it, and never used to pick formats.

## Results (N = 32, FLASH, 1000 seeds)

### P width: the dominant error source

| inputs | pf | dp max | dp mean |
|---|---|---|---|
| soft | 6 | 6.08 | 0.260 |
| soft | 7 | 3.31 | 0.135 |
| soft | **8** | **1.57** | **0.070** |
| peaky | 7 | 2.05 | 0.125 |
| peaky | **8** | **1.26** | **0.069** |
| sharp | 7 | 2.38 | 0.076 |
| sharp | **8** | **1.05** | **0.042** |

Each extra bit of `P` roughly halves the mean error. Going to 9 bits (pf=8, UQ1.8, `256` = 1.0) costs one
extra bit in the MM2 operand and doubles `P`'s HBM element size in NAIVE (2 B instead of 1 B), and halves the error
versus pf=7. That is the trade this project takes, and the larger NAIVE traffic is counted honestly.
pf=7 would keep `P` in one byte but its worst case (3.3 on soft inputs) is well beyond one INT8 step.

### Rounding: almost irrelevant

| (peaky) | dp max | dp mean |
|---|---|---|
| all truncate | 1.260 | 0.0691 |
| round exp2 only | 1.260 | 0.0685 |
| round Z only | 1.260 | 0.0690 |
| round accumulator rescale only | 1.260 | 0.0682 |
| round everything | 1.260 | 0.0677 |

Rounding everything buys 2% of the mean error and nothing at the max, so the datapath truncates. The two
exceptions (`P` from UQ1.15 and the final `out`) round because each is a single adder at a boundary where the
result either feeds MM2 or leaves the chip.

### Reciprocal

| Newton-Raphson steps | dp max | dp mean |
|---|---|---|
| 0 (LUT64 seed only) | 1.437 | 0.1264 |
| **1** | **1.260** | **0.0691** |
| 2 | 1.260 | 0.0692 |

One step is needed (the LUT64 seed is only good to about 7 bits) and a second does nothing.

### Extra fraction bits on `l` and `O`

| `LF = OF` | dp mean |
|---|---|
| 2 | 0.0723 |
| **4** | **0.0691** |
| 6 | 0.0684 |
| 8 | 0.0683 |

Four bits are enough; the accumulators stay at 18 and 26 bits instead of 22 and 30.

### `cfg_scale` fractional bits

| inputs | sf = 12 | sf = 16 (UQ8.16) | **sf = 20 (UQ4.20)** | sf = 24 |
|---|---|---|---|---|
| small, dp max | scale rounds to 0 | 1.143 | **0.080** | 0.056 |
| soft, dp max | 19.8 | 2.41 | **1.57** | 1.57 |
| sharp, dp max | 2.80 | 1.05 | **1.05** | 1.05 |

The original plan's UQ8.16 wastes 8 integer bits: the scale is `sq · sk · 0.72`, normally far below 1, so its
resolution is coarse for small inputs. UQ4.20 fixes that and still reaches 16.
(The `sf = 12` row in the raw file prints an "overflow" error from the earlier overflow check; the host now accepts
a scale that rounds to 0, which just means uniform attention.)

### Sequence length and causal masking (default format)

| N | dp max (bidirectional / causal) | dp mean (bi / causal) |
|---|---|---|
| 4 | 0.62 / 0.59 | 0.067 / 0.042 |
| 8 | 0.85 / 0.85 | 0.064 / 0.049 |
| 16 | 1.06 / 0.98 | 0.066 / 0.053 |
| 32 | 1.26 / 1.02 | 0.069 / 0.056 |

Max error grows slowly with N (more terms in each sum); the mean stays flat.

### FLASH vs NAIVE

| (peaky, N = 32) | dp max | dp mean |
|---|---|---|
| FLASH | 1.260 | 0.069 |
| NAIVE | 1.744 | 0.099 |

FLASH is more accurate, the opposite of what the first plan assumed. NAIVE quantizes every `P` against the global
row max, so small probabilities land on few quantization steps. FLASH quantizes each tile against the running
max, which is usually lower, so early tiles keep more bits and are only rescaled afterwards. The two outputs
never differ by more than 1.3 V-LSB (1000 seeds).

## What the tests pin

`sim/golden/test_golden.py` (52 tests, `SEEDS=1000` for the full sweep):

| Test | Bound |
|---|---|
| `exp2` over all 65 537 Q9.8 inputs ≤ 0 | within 1.5 LSB of UQ1.15; monotone; exactly 0 at shift ≥ 16 |
| `recip` over l = 1..4095 and a sparse sweep to 2^22 | relative error < 2^-13 |
| FLASH accuracy, N ∈ {4,8,16,32} × causal | max < 1.5 V-LSB, mean < 0.09 |
| NAIVE accuracy, N = 32 | max < 2.0, mean < 0.13 |
| FLASH vs NAIVE | max difference < 2.2 V-LSB |
| accumulator widths (adversarial inputs) | `l` < 2^18, `|O|` < 2^25, `|S|` ≤ 64 516, `P` ≤ 256 |
| a fixed 8×4 input → fixed int output | exact (`PINNED_FLASH_8`) |
| HBM bytes, both schedules, every N × causal × `qrows` | equal to the closed form in `traffic.py` |
| `qrows` | changes traffic, never the output |
| presets | deterministic; the toy sentence's "it" attends to "cat" (float and fixed); previous-token preset |

## Open questions carried into later phases

- The `tot` error (up to 12 V-LSB for peaky inputs) is the INT8 Q/K quantization. A per-row or per-block scale
  would reduce it. It is out of scope for v1 and worth a line in the docs page.
- 9-bit `P` makes MM2's A operand unsigned 9-bit × signed 8-bit. The PE must support that (Phase 2).
