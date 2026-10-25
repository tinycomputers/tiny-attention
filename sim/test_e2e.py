"""End-to-end engine test through tiny_attn_top, against the golden model and the closed forms.

Per run, checked exactly:
  * O (every value, int16 Q8.8)     == fixed.flash (FLASH) or fixed.naive (NAIVE), bit for bit
  * HBM read / write bytes          == traffic.py for the mode
  * tiles computed                  == traffic.py tile pairs
  * K/V tile loads                  == kv_tile_loads (FLASH) or tile pairs (NAIVE: one K load per pair)
  * busy cycles                     == sim/golden/schedule.py, FLASH without overlap only (the other cases are
                                       pinned baselines in sim/perf_baseline.json, see test_perf.py)

SEEDS (env, default 200): every seed runs every N in {4..32} x causal {off, on}, with random mode (FLASH or NAIVE),
random overlap (FLASH), random cfg_qrows (<= N), cfg_hbm_latency in 0..64 and random inputs.
qrows_sweep covers every cfg_qrows value and both overlap settings.
"""

import os
import random

import cocotb
import numpy as np

from tbutil import fixed, next_cycle, pack, reset, sample, start_clock  # sets up the golden-model import path
from schedule import cycles as sched_cycles
from traffic import kv_tile_loads, traffic

SEEDS = int(os.environ.get("SEEDS", "200"))
NS = list(range(4, 33, 4))
Q_BASE, K_BASE, V_BASE, O_BASE = 0, 32, 64, 96
MAX_CYCLES = 400_000


async def idle(dut):
    for sig in ("start", "ld_en", "cfg_causal", "cfg_naive", "cfg_overlap"):
        getattr(dut, sig).value = 0
    dut.ld_addr.value = 0
    dut.ld_word.value = 0
    dut.o_rd_addr.value = 0


def make_inputs(rng, n, amp):
    q = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    k = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    v = np.array([[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)])
    return q, k, v


async def host_load(dut, qz, n):
    words = [(Q_BASE + i, pack(qz.q[i], 8)) for i in range(n)]
    words += [(K_BASE + i, pack(qz.k[i], 8)) for i in range(n)]
    words += [(V_BASE + i, pack(qz.v[i], 8)) for i in range(n)]
    for addr, w in words:
        dut.ld_en.value = 1
        dut.ld_addr.value = addr
        dut.ld_word.value = w
        await next_cycle(dut)
    dut.ld_en.value = 0


async def run_one(dut, qz, n, causal, qrows, latency, naive=False, overlap=False):
    mode = "NAIVE" if naive else "FLASH"
    qblock = 4 if naive else qrows
    dut.cfg_nb.value = n // 4
    dut.cfg_gg.value = qblock // 4
    dut.cfg_causal.value = int(causal)
    dut.cfg_naive.value = int(naive)
    dut.cfg_overlap.value = int(overlap and not naive)
    dut.cfg_hbm_latency.value = latency
    dut.cfg_scale.value = qz.cfg_scale
    await host_load(dut, qz, n)

    dut.start.value = 1
    await next_cycle(dut)
    dut.start.value = 0
    for _ in range(MAX_CYCLES):
        await sample(dut)
        if int(dut.done.value):
            break
        await next_cycle(dut)
    else:
        raise AssertionError("engine never finished")
    await next_cycle(dut)  # the DONE cycle is counted; read the counters after it

    ctx = f"{mode} N={n} causal={causal} qrows={qblock} L={latency} overlap={int(overlap and not naive)}"
    t = traffic(n, causal, mode, qblock=qblock)
    assert int(dut.hbm_rd_bytes.value) == t["rd"], f"{ctx}: read bytes {int(dut.hbm_rd_bytes.value)} != {t['rd']}"
    assert int(dut.hbm_wr_bytes.value) == t["wr"], f"{ctx}: write bytes {int(dut.hbm_wr_bytes.value)} != {t['wr']}"
    assert int(dut.tiles.value) == t["tiles"], f"{ctx}: tiles {int(dut.tiles.value)} != {t['tiles']}"
    if naive:
        kv = t["tiles"]  # one K tile load per tile pair (pass 1)
    else:
        kv = kv_tile_loads(n, causal, qrows)
    assert int(dut.kv_loads.value) == kv, f"{ctx}: K/V loads {int(dut.kv_loads.value)} != {kv}"
    if not naive and not overlap:
        want_cycles = sched_cycles(n, causal, qrows, latency)
        assert int(dut.cycles.value) == want_cycles, f"{ctx}: busy cycles {int(dut.cycles.value)} != model {want_cycles}"

    got = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        for half in (0, 1):
            dut.o_rd_addr.value = O_BASE + 2 * i + half
            await sample(dut)
            w = int(dut.o_rd_word.value)
            for lane in range(2):
                v = (w >> (16 * lane)) & 0xFFFF
                got[i, 2 * half + lane] = v - 65536 if v >= 32768 else v
            await next_cycle(dut)
    want = (fixed.naive(qz, n, causal) if naive else fixed.flash(qz, n, causal, qblock=qrows)).out.astype(np.int64)
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)[0]
        raise AssertionError(
            f"{ctx}: O differs at {tuple(int(b) for b in bad)}: expected {want[tuple(bad)]} got {got[tuple(bad)]}"
        )
    return int(dut.cycles.value)


async def setup(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)


@cocotb.test()
async def qrows_sweep(dut):
    """Every cfg_qrows x N x causal, FLASH with and without overlap, plus NAIVE. Two seeds each."""
    await setup(dut)
    rng = random.Random(51)
    for _ in range(2):
        for n in NS:
            for causal in (False, True):
                qz = fixed.quantize(*make_inputs(rng, n, 2.5))
                for qrows in [q for q in (4, 8, 16, 32) if q <= n]:
                    for overlap in (False, True):
                        await run_one(dut, qz, n, causal, qrows, rng.randint(0, 64), overlap=overlap)
                await run_one(dut, qz, n, causal, 4, rng.randint(0, 64), naive=True)


@cocotb.test()
async def random_seeds(dut):
    """SEEDS seeds x every N x causal, random mode, overlap, cfg_qrows (<= N), latency 0..64 and amplitude."""
    await setup(dut)
    rng = random.Random(52)
    for _ in range(SEEDS):
        for n in NS:
            for causal in (False, True):
                naive = rng.random() < 0.35
                qrows = rng.choice([q for q in (4, 8, 16, 32) if q <= n])
                overlap = rng.random() < 0.5
                qz = fixed.quantize(*make_inputs(rng, n, rng.choice([0.5, 1.5, 2.5, 3.5])))
                await run_one(dut, qz, n, causal, qrows, rng.randint(0, 64), naive=naive, overlap=overlap)


def _parse_flash_key(key):
    fields = key.split("/")
    n = int(fields[1][1:])
    causal = bool(int(fields[2][1:]))
    qrows = int(fields[3][1:])
    latency = int(fields[5][1:])
    return n, causal, qrows, latency


PERF_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_baseline.json")


@cocotb.test()
async def perf_grid(dut):
    """Busy cycles over a grid of N x causal x mode x overlap x latency. Pinned in sim/perf_baseline.json.

    PERF_RECORD=1 rewrites the baseline. Otherwise every value must equal the baseline, and overlap must never be
    slower than the sequential schedule for the same configuration.
    """
    import json

    await setup(dut)
    rng = random.Random(53)
    measured = {}
    for n in (8, 16, 32):
        for causal in (False, True):
            qz = fixed.quantize(*make_inputs(rng, n, 2.5))
            for qrows in (4, 16):
                if qrows > n:
                    continue
                for overlap in (False, True):
                    for latency in (0, 10, 40):
                        key = f"flash/N{n}/c{int(causal)}/q{qrows}/o{int(overlap)}/L{latency}"
                        measured[key] = await run_one(dut, qz, n, causal, qrows, latency, overlap=overlap)
            for latency in (0, 10, 40):
                key = f"naive/N{n}/c{int(causal)}/L{latency}"
                measured[key] = await run_one(dut, qz, n, causal, 4, latency, naive=True)

    for key, cyc in measured.items():
        if key.startswith("flash/") and "/o0/" in key:
            n, causal, qrows, latency = _parse_flash_key(key)
            assert cyc == sched_cycles(n, causal, qrows, latency), f"{key}: {cyc} != schedule model"
        if "/o1/" in key:
            seq = measured[key.replace("/o1/", "/o0/")]
            assert cyc <= seq, f"{key}: overlap {cyc} slower than sequential {seq}"

    if os.environ.get("PERF_RECORD"):
        with open(PERF_BASELINE, "w") as f:
            json.dump(dict(sorted(measured.items())), f, indent=1)
            f.write("\n")
        return
    with open(PERF_BASELINE) as f:
        baseline = json.load(f)
    assert set(baseline) == set(measured), "perf grid changed: re-record with PERF_RECORD=1"
    diffs = {k: (baseline[k], measured[k]) for k in measured if baseline[k] != measured[k]}
    assert not diffs, f"busy cycles changed (baseline, measured): {dict(list(diffs.items())[:8])}"
