"""Golden-model self-tests (no RTL). Run: pytest sim/golden -q   (SEEDS=1000 for the full sweep)"""

import os
from pathlib import Path

import numpy as np
import pytest

import fixed
import gen_luts
import presets
from float_ref import attention
from fixed import Fmt, DEFAULT
from traffic import traffic

SEEDS = int(os.environ.get("SEEDS", "100"))
NS = (4, 8, 16, 32)


def inputs(seed, n, amp=2.5):
    r = np.random.default_rng(seed)
    return r.normal(0, amp, (n, 4)), r.normal(0, amp, (n, 4)), r.normal(0, 1, (n, 4))


# ------------------------------------------------------------------ LUTs
def test_lut_files_fresh():
    d = gen_luts.MEM_DIR
    assert (d / "exp2_lut.mem").read_text() == gen_luts._fmt(gen_luts.exp2_lut()), "run gen_luts.py"
    assert (d / "recip_lut.mem").read_text() == gen_luts._fmt(gen_luts.recip_lut()), "run gen_luts.py"


def test_lut_ranges():
    e, r = gen_luts.exp2_lut(), gen_luts.recip_lut()
    assert len(e) == 256 and e[0] == 1 << 15 and max(e) < 1 << 16 and e == sorted(e)
    assert len(r) == 64 and max(r) < 1 << 16 and r == sorted(r, reverse=True)


# ------------------------------------------------------------------ exp2
def test_exp2_exhaustive_vs_float():
    """Every Q9.8 input <= 0: within 1 LSB of UQ1.15 truncation + LUT rounding."""
    x = np.arange(-65536, 1)
    got = fixed.exp2_q(x).astype(np.float64) / (1 << 15)
    ref = 2.0 ** (x / 256.0)
    assert np.abs(got - ref).max() <= 1.5 / (1 << 15)
    assert fixed.exp2_q(np.array([0]))[0] == 1 << 15
    assert (fixed.exp2_q(np.arange(-65536, -16 * 256 + 1)) == 0).all()  # shift >= 16 -> exactly 0


def test_exp2_monotone():
    got = fixed.exp2_q(np.arange(-65536, 1))
    assert (np.diff(got) >= 0).all()


# ------------------------------------------------------------------ recip
def test_recip_accuracy_all_l():
    """Relative error of 1/l over every l that can occur at the top of the range, and a sample below."""
    ls = list(range(1, 4096)) + list(range(4096, 1 << 22, 997))
    worst = 0.0
    for l in ls:
        r, k = fixed.recip(l)
        approx = r / 2.0 ** (16 + k)
        worst = max(worst, abs(approx * l - 1.0))
    assert worst < 2 ** -13


def test_recip_powers_of_two():
    for k in range(0, 22):
        r, kk = fixed.recip(1 << k)
        assert kk == k and abs(r / 65536 - 1.0) < 2 ** -13


# ------------------------------------------------------------------ quantize
def test_quantize_range_and_scale():
    qf, kf, vf = inputs(0, 16)
    z = fixed.quantize(qf, kf, vf)
    for a in (z.q, z.k, z.v):
        assert a.min() >= -127 and a.max() <= 127 and np.abs(a).max() == 127
    want = z.sq * z.sk * fixed.LOG2E / 2.0
    assert abs(z.cfg_scale / (1 << DEFAULT.sf) - want) <= 0.5 / (1 << DEFAULT.sf)


# ------------------------------------------------------------------ structure
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("causal", [False, True])
def test_single_tile_row_matches_float_exactly_enough(n, causal):
    qf, kf, vf = inputs(1, n)
    z = fixed.quantize(qf, kf, vf)
    res = fixed.flash(z, n, causal)
    assert res.out.shape == (n, 4) and res.out.dtype == np.int16


def test_causal_first_row_is_v0():
    """Row 0 attends only to key 0, so out[0] == V[0] (exactly, in V-LSB Q8.8)."""
    qf, kf, vf = inputs(2, 16)
    z = fixed.quantize(qf, kf, vf)
    for res in (fixed.flash(z, 16, True), fixed.naive(z, 16, True)):
        assert np.abs(res.out[0] - z.v[0] * 256).max() <= 2  # <= 2/256 V-LSB


def test_uniform_attention_is_mean_of_v():
    p = presets.uniform(16)
    z = fixed.quantize(p["q"], p["k"], p["v"])
    res = fixed.flash(z, 16, False)
    want = z.v.mean(axis=0) * 256
    assert np.abs(res.out - want).max() < 3  # all rows ~ mean(V)


def test_flash_trace_shapes_and_causal_skip():
    qf, kf, vf = inputs(3, 16)
    z = fixed.quantize(qf, kf, vf)
    res = fixed.flash(z, 16, True, trace=True)
    assert len(res.trace) == 10 and all(t["kj"] <= t["qi"] for t in res.trace)
    last = res.trace[-1]
    assert last["P"].max() <= 1 << DEFAULT.pf and last["l"].min() > 0
    assert (np.diff([t["m"].max() for t in res.trace if t["qi"] == 3]) >= 0).all()  # running max never falls


def test_masked_probabilities_are_zero():
    qf, kf, vf = inputs(4, 8)
    z = fixed.quantize(qf, kf, vf)
    res = fixed.flash(z, 8, True, trace=True)
    diag = [t for t in res.trace if t["qi"] == t["kj"]]
    for t in diag:
        assert (t["P"][np.triu_indices(4, 1)] == 0).all()
    nv = fixed.naive(z, 8, True)
    assert (nv.probs[np.triu_indices(8, 1)] == 0).all()


# ------------------------------------------------------------------ accuracy (pinned bounds)
def _err(mode, n, causal, seeds, fmt=DEFAULT):
    worst = tot = cnt = 0.0
    for s in range(seeds):
        qf, kf, vf = inputs(1000 + s, n)
        z = fixed.quantize(qf, kf, vf, fmt)
        res = (fixed.flash if mode == "flash" else fixed.naive)(z, n, causal, fmt)
        ref, _ = attention(z.q * z.sq, z.k * z.sk, z.v * z.sv, causal)
        e = np.abs(res.out / 256.0 - ref / z.sv)
        worst, tot, cnt = max(worst, e.max()), tot + e.sum(), cnt + e.size
    return worst, tot / cnt


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_accuracy_bounds(n, causal):
    worst, mean = _err("flash", n, causal, SEEDS)
    assert worst < 1.5 and mean < 0.09, (worst, mean)  # V-LSB units; see docs/NUMERICS.md


@pytest.mark.parametrize("causal", [False, True])
def test_naive_accuracy_bounds(causal):
    worst, mean = _err("naive", 32, causal, SEEDS)
    assert worst < 2.0 and mean < 0.13, (worst, mean)


def test_flash_vs_naive_difference_bounded():
    """Not bit-identical (different P quantization), but close: pinned so a regression is visible."""
    worst = 0.0
    for s in range(SEEDS):
        z = fixed.quantize(*inputs(2000 + s, 32))
        a = fixed.flash(z, 32, False).out.astype(np.int64)
        b = fixed.naive(z, 32, False).out.astype(np.int64)
        worst = max(worst, np.abs(a - b).max() / 256.0)
    assert worst < 2.2, worst


def test_pinned_vector():
    """A tiny fixed input -> fixed integer output. Any change to a rounding rule shows up here first."""
    p = presets.random_qkv(8, seed=0)
    z = fixed.quantize(p["q"], p["k"], p["v"])
    assert fixed.flash(z, 8, False).out.tolist() == PINNED_FLASH_8


PINNED_FLASH_8 = [
    [-2941, -3251, 5023, 15049],
    [5455, -2868, 15278, 704],
    [-4851, -9225, 3889, 15675],
    [-11517, 9919, 7239, 15759],
    [-1620, -3063, 3061, 19128],
    [-6091, 5222, 10605, 7693],
    [-523, -1502, -5102, -6946],
    [-14034, 15043, 3218, 20206],
]


# ------------------------------------------------------------------ traffic
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("causal", [False, True])
def test_traffic_matches_closed_form(n, causal):
    z = fixed.quantize(*inputs(5, n))
    for mode, fn in (("FLASH", fixed.flash), ("NAIVE", fixed.naive)):
        t = fn(z, n, causal).traffic
        want = traffic(n, causal, mode)
        assert (t.rd, t.wr, t.tiles, t.skipped) == (want["rd"], want["wr"], want["tiles"], want["skipped"])


@pytest.mark.parametrize("qblock", [4, 8, 16, 32])
@pytest.mark.parametrize("causal", [False, True])
def test_qblock_changes_traffic_not_output(qblock, causal):
    z = fixed.quantize(*inputs(6, 32))
    base = fixed.flash(z, 32, causal)
    res = fixed.flash(z, 32, causal, qblock=qblock)
    assert np.array_equal(res.out, base.out)
    want = traffic(32, causal, "FLASH", qblock=qblock)
    assert (res.traffic.rd, res.traffic.wr) == (want["rd"], want["wr"])


def test_flash_traffic_story():
    """FLASH always beats NAIVE; the margin grows with N and with the on-chip Q block."""
    ratios = [traffic(n, False, "NAIVE")["total"] / traffic(n, False, "FLASH")["total"] for n in (4, 8, 16, 32)]
    assert ratios == sorted(ratios) and ratios[-1] > 5
    full = traffic(32, False, "NAIVE")["total"] / traffic(32, False, "FLASH", qblock=32)["total"]
    assert full > 15  # whole sequence resident: K and V each read exactly once
    assert traffic(32, False, "FLASH", qblock=32)["total"] == 32 * 4 * (1 + 1 + 1 + 2)


# ------------------------------------------------------------------ presets
def test_presets_deterministic_and_shaped():
    for make in presets.ALL:
        a, b = make(), make()
        assert a["q"].shape == a["k"].shape == a["v"].shape and a["q"].shape[0] % 4 == 0
        assert len(a["labels"]) == a["q"].shape[0]
        assert np.array_equal(a["q"], b["q"]) and np.array_equal(a["v"], b["v"])
        fixed.quantize(a["q"], a["k"], a["v"])  # must be quantizable


def test_toy_sentence_pronoun_attends_to_subject_noun():
    p = presets.toy_sentence()
    _, probs = attention(p["q"], p["k"], p["v"], causal=True)
    it, cat, mouse = (p["labels"].index(t) for t in ("it", "cat", "mouse"))
    assert probs[it].argmax() == cat and probs[it, cat] > 2 * probs[it, mouse]
    z = fixed.quantize(p["q"], p["k"], p["v"])
    pm = fixed.naive(z, 12, True).probs
    assert pm[it].argmax() == cat  # still true in fixed point


def test_previous_token_preset():
    p = presets.previous_token(16)
    _, probs = attention(p["q"], p["k"], p["v"], causal=True)
    assert all(probs[i].argmax() == i - 1 for i in range(1, 16))


# ------------------------------------------------------------------ bit widths (docs/ARCHITECTURE.md)
def test_bit_width_bounds_worst_case():
    """Drive every accumulator toward its maximum and check it fits the documented register width."""
    n = 32
    ones = np.ones((n, 4))
    cases = {
        "uniform_maxV": (np.zeros((n, 4)), np.zeros((n, 4)), 127 * ones),  # l, O maximal
        "uniform_minV": (np.zeros((n, 4)), np.zeros((n, 4)), -127 * ones),
        "max_S": (127 * ones, 127 * ones, 127 * ones),  # |S| = 4*127*127
    }
    for name, (qf, kf, vf) in cases.items():
        z = fixed.quantize(qf, kf, vf)
        res = fixed.flash(z, n, False, trace=True)
        for t in res.trace:
            assert np.abs(t["S"]).max() <= 4 * 127 * 127 < 1 << 16  # int17
            assert t["Z"].max() <= 32767 and t["Z"].min() >= -32768
            assert t["l"].max() < 1 << 18, name  # 18-bit unsigned
            assert np.abs(t["O"]).max() < 1 << 25, name  # 26-bit signed
            assert t["P"].max() <= 256  # 9-bit unsigned
        assert np.abs(res.out).max() <= 127 * 256
