"""Numerics study: sweep format knobs and report error vs float. Run: python sim/golden/study.py [seeds]

Errors are in V-LSB units (1.0 = one INT8 step of V), so they compare directly with the inherent
rounding of an INT8 result. `dp` = datapath error vs float attention on the *dequantized* INT8
inputs (what the hardware is responsible for); `tot` = vs float on the original inputs.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

import fixed
from float_ref import attention


def gen_inputs(rng, n, kind):
    """Gaussian Q/K with std `amp`: soft/peaky/sharp attention; `small` has tiny magnitudes (stresses cfg_scale)."""
    amp = {"soft": 1.5, "peaky": 2.5, "sharp": 3.5, "small": 0.3}[kind]
    q = rng.normal(0, amp, (n, fixed.D))
    k = rng.normal(0, amp, (n, fixed.D))
    v = rng.normal(0, 1.0, (n, fixed.D))
    return q, k, v


def errors(fmt, n, causal, kind, seeds, mode="flash", seed0=0):
    dp_max = dp_sum = tot_max = tot_sum = 0.0
    cnt = 0
    p_max = 0.0
    for s in range(seeds):
        rng = np.random.default_rng(seed0 + s)
        qf, kf, vf = gen_inputs(rng, n, kind)
        qz = fixed.quantize(qf, kf, vf, fmt)
        res = (fixed.flash if mode == "flash" else fixed.naive)(qz, n, causal, fmt)
        got = res.out.astype(np.float64) / 256.0  # V-LSB units
        ref_dq, _ = attention(qz.q * qz.sq, qz.k * qz.sk, qz.v * qz.sv, causal)
        ref_tot, _ = attention(qf, kf, vf, causal)
        e_dp = np.abs(got - ref_dq / qz.sv)
        e_tot = np.abs(got - ref_tot / qz.sv)
        dp_max = max(dp_max, e_dp.max())
        tot_max = max(tot_max, e_tot.max())
        dp_sum += e_dp.sum()
        tot_sum += e_tot.sum()
        cnt += e_dp.size
    return dict(dp_max=dp_max, dp_mean=dp_sum / cnt, tot_max=tot_max, tot_mean=tot_sum / cnt)


def row(label, e):
    return f"{label:<34} dp max {e['dp_max']:6.3f} mean {e['dp_mean']:6.4f} | tot max {e['tot_max']:6.3f} mean {e['tot_mean']:6.4f}"


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    base = fixed.DEFAULT
    print(f"== P width sweep (N=32, FLASH, {seeds} seeds per cell)")
    for kind in ("soft", "peaky", "sharp"):
        for pf in (5, 6, 7, 8):
            print(row(f"{kind:5} pf={pf}", errors(replace(base, pf=pf), 32, False, kind, seeds)))
    print("== rounding knobs (peaky, N=32, exp/z/acc round vs truncate)")
    for name, kw in [
        ("all truncate (default)", {}),
        ("exp round", dict(exp_round=True)),
        ("z round", dict(z_round=True)),
        ("acc round", dict(acc_round=True)),
        ("all round", dict(exp_round=True, z_round=True, acc_round=True)),
    ]:
        print(row(name, errors(replace(base, **kw), 32, False, "peaky", seeds)))
    print("== recip refinement (peaky, N=32)")
    for nr in (0, 1, 2):
        print(row(f"nr_steps={nr}", errors(replace(base, nr_steps=nr), 32, False, "peaky", seeds)))
    print("== l/O extra fractional bits (peaky, N=32)")
    for b in (2, 4, 6, 8):
        print(row(f"lf=of={b}", errors(replace(base, lf=b, of=b), 32, False, "peaky", seeds)))
    print("== cfg_scale fractional bits sf (cfg_scale is 24 bits; N=32)")
    for kind in ("small", "soft", "sharp"):
        for sf in (12, 16, 20, 24):
            try:
                print(row(f"{kind:5} sf={sf}", errors(replace(base, sf=sf), 32, False, kind, seeds)))
            except ValueError as e:
                print(f"{kind:5} sf={sf:<2}  overflow: {e}")
    print("== sequence length and causal (default fmt)")
    for n in (4, 8, 16, 32):
        for causal in (False, True):
            print(row(f"N={n:2} causal={int(causal)} flash", errors(base, n, causal, "peaky", seeds)))
    print("== FLASH vs NAIVE (peaky, default fmt)")
    for mode in ("flash", "naive"):
        print(row(f"{mode} N=32", errors(base, 32, False, "peaky", seeds, mode)))


if __name__ == "__main__":
    main()
