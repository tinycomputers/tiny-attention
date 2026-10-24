"""Single source of truth for the exp2 and reciprocal ROMs.

fixed.py imports exp2_lut()/recip_lut() from here; running this file writes the same values to
rtl/mem/*.mem for $readmemh. test_golden.py fails if the checked-in files drift.

    python sim/golden/gen_luts.py
"""

from pathlib import Path

MEM_DIR = Path(__file__).resolve().parents[2] / "rtl" / "mem"

EXP2_FRAC_BITS = 8  # LUT256: index = fractional part of the Q9.8 argument
EXP2_OUT_FRAC = 15  # entries are UQ1.15
RECIP_IDX_BITS = 6  # LUT64: index = top 6 fraction bits of the normalized mantissa
RECIP_OUT_FRAC = 16  # entries are UQ0.16


def exp2_lut() -> list[int]:
    """LUT[f] = round(2**15 * 2**(f/256)) for f in 0..255. Range 32768..65358, so 16 bits."""
    return [round((1 << EXP2_OUT_FRAC) * 2 ** (f / (1 << EXP2_FRAC_BITS))) for f in range(1 << EXP2_FRAC_BITS)]


def recip_lut() -> list[int]:
    """LUT[i] = round(2**16 / (1 + (i + 0.5) / 64)): seed for 1/M, M in [1, 2). Range 32897..65028, 16 bits."""
    n = 1 << RECIP_IDX_BITS
    return [round((1 << RECIP_OUT_FRAC) / (1 + (i + 0.5) / n)) for i in range(n)]


def _fmt(values: list[int]) -> str:
    return "".join(f"{v:04x}\n" for v in values)


def write_all() -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    (MEM_DIR / "exp2_lut.mem").write_text(_fmt(exp2_lut()))
    (MEM_DIR / "recip_lut.mem").write_text(_fmt(recip_lut()))


if __name__ == "__main__":
    write_all()
    print(f"wrote {MEM_DIR}/exp2_lut.mem (256) and recip_lut.mem (64)")
