"""End-to-end FLASH through tiny_attn_top, against the golden model and the closed forms.

Per run, checked exactly:
  * O (every value, int16 Q8.8)     == fixed.flash(qblock=cfg_qrows)  (bit-exact)
  * HBM read bytes / write bytes    == traffic.py (FLASH)
  * tiles computed, K/V tile loads  == traffic.py / kv_tile_loads
  * busy cycles                     == sim/golden/schedule.py (the controller's state sequence)

SEEDS (env, default 200): each seed runs every sequence length N in {4..32} x causal {off, on}, with random
cfg_qrows (<= N), cfg_hbm_latency in 0..64 and random inputs. qrows_sweep covers every cfg_qrows value.
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
    for sig in ("start", "ld_en", "cfg_causal"):
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


async def run_one(dut, qz, n, causal, qrows, latency):
    dut.cfg_nb.value = n // 4
    dut.cfg_gg.value = qrows // 4
    dut.cfg_causal.value = int(causal)
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

    ctx = f"N={n} causal={causal} qrows={qrows} L={latency}"
    t = traffic(n, causal, "FLASH", qblock=qrows)
    assert int(dut.hbm_rd_bytes.value) == t["rd"], f"{ctx}: read bytes {int(dut.hbm_rd_bytes.value)} != {t['rd']}"
    assert int(dut.hbm_wr_bytes.value) == t["wr"], f"{ctx}: write bytes {int(dut.hbm_wr_bytes.value)} != {t['wr']}"
    assert int(dut.tiles.value) == t["tiles"], f"{ctx}: tiles {int(dut.tiles.value)} != {t['tiles']}"
    kv = kv_tile_loads(n, causal, qrows)
    assert int(dut.kv_loads.value) == kv, f"{ctx}: K/V loads {int(dut.kv_loads.value)} != {kv}"
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
    want = fixed.flash(qz, n, causal, qblock=qrows).out.astype(np.int64)
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)[0]
        raise AssertionError(
            f"{ctx}: O differs at {tuple(int(b) for b in bad)}: expected {want[tuple(bad)]} got {got[tuple(bad)]}"
        )


async def setup(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)


@cocotb.test()
async def qrows_sweep(dut):
    """Every cfg_qrows value x every N x causal, two seeds each. The output must not depend on qrows."""
    await setup(dut)
    rng = random.Random(51)
    for _ in range(2):
        for n in NS:
            for causal in (False, True):
                qz = fixed.quantize(*make_inputs(rng, n, 2.5))
                for qrows in [q for q in (4, 8, 16, 32) if q <= n]:
                    await run_one(dut, qz, n, causal, qrows, rng.randint(0, 64))


@cocotb.test()
async def random_seeds(dut):
    """SEEDS seeds x every N x causal, random cfg_qrows (<= N), latency 0..64 and input amplitude."""
    await setup(dut)
    rng = random.Random(52)
    for _ in range(SEEDS):
        for n in NS:
            for causal in (False, True):
                qrows = rng.choice([q for q in (4, 8, 16, 32) if q <= n])
                qz = fixed.quantize(*make_inputs(rng, n, rng.choice([0.5, 1.5, 2.5, 3.5])))
                await run_one(dut, qz, n, causal, qrows, rng.randint(0, 64))
