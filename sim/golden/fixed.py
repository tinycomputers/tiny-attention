"""Bit-exact fixed-point model of the TinyAttention datapath (both schedules).

Every shift and rounding step here must have a one-to-one counterpart in the RTL and an entry in
docs/ARCHITECTURE.md. All integer math is int64 (largest intermediate is ~46 bits).

Conventions: `>>` on int64 is an arithmetic (floor) shift. "round" means add half an LSB, then floor.

Formats (see docs/ARCHITECTURE.md for the bit-width proofs):
  Q, K, V  int8                         S  int20 = Q.K^T
  Z        Q8.8 int16, mask = MASK      m  Q8.8
  exp2 out UQ1.15                       alpha UQ1.15
  P        UQ1.pf (pf=8 -> 9 bits unsigned, max 256 = 1.0)
  l        integer, real_l * 2**(pf+LF)  O  integer, sum(P*V) * 2**OF in V-LSB units
  out      Q8.8 int16 in V-LSB units
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import gen_luts

D = 4
BR = BC = 4
MASK = -32768  # Z value of a masked logit (0x8000); unmasked Z saturates to +-32767
Z_MAX = 32767
M_INIT = -32768
LOG2E = float(np.log2(np.e))

_EXP2 = np.array(gen_luts.exp2_lut(), dtype=np.int64)
_RECIP = np.array(gen_luts.recip_lut(), dtype=np.int64)


@dataclass(frozen=True)
class Fmt:
    """Tunable number-format knobs. DEFAULT is what the RTL implements."""

    pf: int = 8  # fractional bits of P (P max = 1 << pf, so P is 9 bits unsigned)
    lf: int = 4  # extra fractional bits of l
    of: int = 4  # extra fractional bits of O (keep == lf so they cancel in O/l)
    sf: int = 20  # fractional bits of cfg_scale (a 24-bit UQ4.20 register)
    exp_round: bool = False  # exp2 shift: round-to-nearest instead of truncate
    z_round: bool = False  # S*scale -> Z
    acc_round: bool = False  # (l*alpha)>>15 and (O*alpha)>>15
    nr_steps: int = 1  # Newton-Raphson steps in recip (0 = LUT seed only)


DEFAULT = Fmt()


# ---------------------------------------------------------------- host-side quantization
@dataclass
class Quantized:
    q: np.ndarray  # int8 N x D
    k: np.ndarray
    v: np.ndarray
    sq: float
    sk: float
    sv: float
    cfg_scale: int  # sq*sk*log2(e)/sqrt(D) * 2**sf


def quantize(qf, kf, vf, fmt: Fmt = DEFAULT) -> Quantized:
    """Symmetric per-tensor INT8. cfg_scale folds scales, log2(e) and 1/sqrt(D) into one 24-bit register."""

    def one(x):
        x = np.asarray(x, dtype=np.float64)
        s = float(np.abs(x).max()) / 127.0 or 1.0
        return np.clip(np.rint(x / s), -127, 127).astype(np.int64), s

    q, sq = one(qf)
    k, sk = one(kf)
    v, sv = one(vf)
    scale = int(round(sq * sk * LOG2E / np.sqrt(D) * (1 << fmt.sf)))
    # scale == 0 is legal: logits too small to resolve, so every Z is 0 and attention is uniform
    if scale >= (1 << 24):
        raise ValueError(f"cfg_scale {scale} does not fit 24 bits at sf={fmt.sf} (inputs badly scaled)")
    return Quantized(q, k, v, sq, sk, sv, scale)


# ---------------------------------------------------------------- arithmetic units
def exp2_q(x, fmt: Fmt = DEFAULT):
    """2**(x/256) for x <= 0 (Q9.8 int) -> UQ1.15. LUT256 + right shift; exact 0 once the shift >= 16."""
    x = np.asarray(x, dtype=np.int64)
    assert (x <= 0).all(), "exp2_q needs x <= 0"
    s = -(x >> 8)  # integer part of -x, >= 0 (floor of x, so f below is the positive remainder)
    f = x & 0xFF
    lut = _EXP2[f]
    sc = np.minimum(s, 17)  # shifting 16-bit lut by >= 17 is 0 either way; avoids huge shifts
    if fmt.exp_round:
        half = np.where(sc > 0, 1 << np.maximum(sc - 1, 0), 0)
        out = (lut + half) >> sc
        return np.where(s > 16, 0, out)
    return np.where(s >= 16, 0, lut >> sc)


def recip(l, fmt: Fmt = DEFAULT):
    """1/l for integer l >= 1. Returns (r, k): 1/l ~= r / 2**(16 + k), r in UQ1.16 (0.5..1.0 * 2**16)."""
    l = int(l)
    assert l >= 1
    k = l.bit_length() - 1
    m16 = (l << 15) >> k if k <= 15 else l >> (k - 15)  # mantissa in [1,2) as UQ1.15 (truncating)
    r = int(_RECIP[(m16 >> 9) & 63])  # UQ0.16 seed
    for _ in range(fmt.nr_steps):
        t = (m16 * r) >> 15  # M*r ~ 1.0 as UQ1.16
        r = (r * ((2 << 16) - t)) >> 16  # r * (2 - M*r)
    return r, k


def normalize(o, l, fmt: Fmt = DEFAULT):
    """out = round(o / l * 256) as int16 Q8.8 (o, l share the same frac bits). o: int64 array, l: int."""
    r, k = recip(l, fmt)
    sh = 8 + k  # (o * r) / 2**(16+k) * 256
    out = (np.asarray(o, dtype=np.int64) * r + (1 << (sh - 1))) >> sh
    return np.clip(out, -32768, 32767)


def scale_mask(s, scale, valid, fmt: Fmt = DEFAULT):
    """S (int20) -> Z (Q8.8): sat((S*cfg_scale) >> (sf-8)), masked entries -> MASK."""
    prod = np.asarray(s, dtype=np.int64) * scale  # |S| < 2**16, scale < 2**24 -> < 2**40
    sh = fmt.sf - 8
    z = (prod + (1 << (sh - 1))) >> sh if fmt.z_round else prod >> sh
    z = np.clip(z, -Z_MAX, Z_MAX)
    return np.where(valid, z, MASK)


def p_from_exp(e, fmt: Fmt = DEFAULT):
    """UQ1.15 exp2 result -> P (UQ1.pf), 0..2**pf."""
    sh = 15 - fmt.pf
    return (e + (1 << (sh - 1))) >> sh


def mm(a, b):
    """The systolic array: exact integer matmul (int8 x int8 or uint8 x int8)."""
    return np.asarray(a, dtype=np.int64) @ np.asarray(b, dtype=np.int64)


def _acc(x, alpha, fmt):
    prod = x * alpha
    return (prod + (1 << 14)) >> 15 if fmt.acc_round else prod >> 15


def online_softmax_step(z, m, l, fmt: Fmt = DEFAULT):
    """One tile step of the online softmax for R rows. z: R x 4 Q8.8 (masked = MASK), m: R, l: R.
    Returns (m_new, alpha, p, l_new). This is exactly the RTL `online_softmax` module, vectorized over rows."""
    z = np.asarray(z, dtype=np.int64)
    valid = z != MASK
    m_new = np.maximum(m, z.max(axis=1))
    alpha = exp2_q(m - m_new, fmt)  # UQ1.15
    e = exp2_q(np.where(valid, z - m_new[:, None], 0), fmt)
    p = np.where(valid, p_from_exp(e, fmt), 0)  # UQ1.pf
    l_new = _acc(l, alpha, fmt) + (p.sum(axis=1) << fmt.lf)
    return m_new, alpha, p, l_new


def o_update(o, alpha, pv, fmt: Fmt = DEFAULT):
    """O = (O * alpha) >> 15 + (P.V << OF) for R rows. o: R x 4, alpha: R, pv: R x 4. RTL `o_accumulator`."""
    return _acc(np.asarray(o, dtype=np.int64), np.asarray(alpha, dtype=np.int64)[:, None], fmt) + (
        np.asarray(pv, dtype=np.int64) << fmt.of
    )


# ---------------------------------------------------------------- schedules
@dataclass
class Traffic:
    rd: int = 0
    wr: int = 0
    skipped: int = 0  # (qi, kj) tile pairs never computed (causal)
    tiles: int = 0  # tile pairs computed

    @property
    def total(self):
        return self.rd + self.wr


@dataclass
class Result:
    out: np.ndarray  # N x D int16 Q8.8
    traffic: Traffic
    trace: list = field(default_factory=list)  # FLASH: one dict per (qi, kj)
    probs: np.ndarray | None = None  # NAIVE: N x N P matrix (UQ1.pf), 0 where never computed


def _tiles(n, causal):
    nb = n // BR
    for qi in range(nb):
        kjs = range(qi + 1) if causal else range(nb)
        yield qi, kjs


def _valid(qi, kj, causal):
    rows = qi * BR + np.arange(BR)[:, None]
    cols = kj * BC + np.arange(BC)[None, :]
    return (cols <= rows) if causal else np.ones((BR, BC), dtype=bool)


def flash(qz: Quantized, n: int, causal: bool = False, fmt: Fmt = DEFAULT, trace: bool = False,
          qblock: int = BR) -> Result:
    """FlashAttention-2 schedule: Q blocks outer, m/l/O resident on chip, S and P never leave.

    `qblock` = rows resident per Q block (multiple of 4, <= n). Each K/V tile is loaded once per Q block and
    reused by all qblock/4 row groups, so HBM traffic falls as qblock grows. Rows are independent and every
    row still sees its K/V tiles in ascending order, so the output does not depend on qblock.
    """
    assert n % BR == 0 and 4 <= n and qblock % BR == 0 and 4 <= qblock <= n
    out = np.zeros((n, D), dtype=np.int64)
    tr = Traffic()
    log = []
    nb = n // BR
    gpb = qblock // BR  # row groups per block
    for b0 in range(0, nb, gpb):
        groups = list(range(b0, min(b0 + gpb, nb)))
        state = {}
        for qi in groups:
            tr.rd += BR * D  # Q tile, int8
            state[qi] = (np.full(BR, M_INIT, dtype=np.int64), np.zeros(BR, dtype=np.int64),
                         np.zeros((BR, D), dtype=np.int64))
        for kj in (range(groups[-1] + 1) if causal else range(nb)):
            kt = qz.k[kj * BC:(kj + 1) * BC]
            vt = qz.v[kj * BC:(kj + 1) * BC]
            tr.rd += 2 * BC * D  # K and V tiles, loaded once for the whole block
            for qi in groups:
                if causal and kj > qi:
                    continue
                m, l, o = state[qi]
                qt = qz.q[qi * BR:(qi + 1) * BR]
                tr.tiles += 1
                valid = _valid(qi, kj, causal)
                s = mm(qt, kt.T)  # MM1
                z = scale_mask(s, qz.cfg_scale, valid, fmt)
                m, alpha, p, l = online_softmax_step(z, m, l, fmt)
                o = o_update(o, alpha, mm(p, vt), fmt)  # MM2
                state[qi] = (m, l, o)
                if trace:
                    log.append(dict(qi=qi, kj=kj, S=s, Z=z, m=m.copy(), alpha=alpha, P=p, l=l.copy(), O=o.copy()))
        for qi in groups:
            _, l, o = state[qi]
            for r in range(BR):
                out[qi * BR + r] = normalize(o[r], int(l[r]), fmt)
            tr.wr += BR * D * 2  # O tile, int16
    tr.skipped = nb * nb - tr.tiles
    return Result(out.astype(np.int16), tr, log)


def naive(qz: Quantized, n: int, causal: bool = False, fmt: Fmt = DEFAULT) -> Result:
    """Standard (3-pass safe-softmax) attention: Z and P round-trip through HBM. Same datapath as flash()."""
    assert n % BR == 0 and 4 <= n
    out = np.zeros((n, D), dtype=np.int64)
    tr = Traffic()
    hbm_z = np.full((n, n), MASK, dtype=np.int64)  # Q8.8, 2 B per element
    hbm_p = np.zeros((n, n), dtype=np.int64)  # UQ1.pf
    p_bytes = 1 if fmt.pf <= 7 else 2  # default pf=8 -> 9-bit P, stored as 2 B
    nb = n // BR
    # pass 1: S tiles -> Z -> HBM
    for qi, kjs in _tiles(n, causal):
        qt = qz.q[qi * BR:(qi + 1) * BR]
        tr.rd += BR * D
        for kj in kjs:
            kt = qz.k[kj * BC:(kj + 1) * BC]
            tr.rd += BC * D
            tr.tiles += 1
            z = scale_mask(mm(qt, kt.T), qz.cfg_scale, _valid(qi, kj, causal), fmt)
            hbm_z[qi * BR:(qi + 1) * BR, kj * BC:(kj + 1) * BC] = z
            tr.wr += BR * BC * 2
    tr.skipped = nb * nb - tr.tiles
    # pass 2: read Z back for the row max
    m = np.full(n, M_INIT, dtype=np.int64)
    for qi, kjs in _tiles(n, causal):
        for kj in kjs:
            blk = hbm_z[qi * BR:(qi + 1) * BR, kj * BC:(kj + 1) * BC]
            tr.rd += BR * BC * 2
            m[qi * BR:(qi + 1) * BR] = np.maximum(m[qi * BR:(qi + 1) * BR], blk.max(axis=1))
    # pass 3: read Z again, exp2 -> P -> HBM, row sums
    l = np.zeros(n, dtype=np.int64)
    for qi, kjs in _tiles(n, causal):
        rows = slice(qi * BR, (qi + 1) * BR)
        for kj in kjs:
            blk = hbm_z[rows, kj * BC:(kj + 1) * BC]
            tr.rd += BR * BC * 2
            valid = blk != MASK
            e = exp2_q(np.where(valid, blk - m[rows, None], 0), fmt)
            p = np.where(valid, p_from_exp(e, fmt), 0)
            hbm_p[rows, kj * BC:(kj + 1) * BC] = p
            tr.wr += BR * BC * p_bytes
            l[rows] += p.sum(axis=1) << fmt.lf
    # pass 4: P.V, normalize
    for qi, kjs in _tiles(n, causal):
        rows = slice(qi * BR, (qi + 1) * BR)
        o = np.zeros((BR, D), dtype=np.int64)
        for kj in kjs:
            p = hbm_p[rows, kj * BC:(kj + 1) * BC]
            tr.rd += BR * BC * p_bytes + BC * D  # P tile + V tile
            o += mm(p, qz.v[kj * BC:(kj + 1) * BC]) << fmt.of
        for r in range(BR):
            out[qi * BR + r] = normalize(o[r], int(l[qi * BR + r]), fmt)
        tr.wr += BR * D * 2
    return Result(out.astype(np.int16), tr, [], hbm_p)


def dequant_out(res: Result, qz: Quantized):
    """Q8.8 V-LSB output -> real values (host-side, derived)."""
    return res.out.astype(np.float64) / 256.0 * qz.sv
