"""attn_tile lockstep: one Q block against its K/V tiles, compared with fixed.flash(trace=True) at EVERY tile
boundary (S, Z, P, alpha, m, l, O) and on the final normalized rows. Stops at the first divergence with a readable diff.
"""

import random

import cocotb
import numpy as np

from tbutil import fixed, next_cycle, pack, reset, sample, start_clock, unpack

TILE_CYCLES = 20  # tile_start asserted -> tile_done visible (documented in rtl/attn_tile.sv)


def tile_matrix(x, w, n):
    """flat dbg vector of 16 elements -> 4x4 list of lists (row-major)"""
    flat = unpack(x, w, n * n)
    return [flat[r * n:(r + 1) * n] for r in range(n)]


async def idle(dut):
    for sig in ("blk_clear", "q_load_valid", "kv_load_valid", "tile_start", "tile_diag", "tile_last"):
        getattr(dut, sig).value = 0
    for sig in ("q_load_idx", "q_load_row", "kv_load_idx", "kt_row", "v_row"):
        getattr(dut, sig).value = 0
    dut.cfg_scale.value = 0


def diff(name, got, want, ctx):
    got, want = np.array(got), np.array(want)
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)[0]
        raise AssertionError(f"{ctx}: {name} differs at {tuple(bad)}: expected {want[tuple(bad)]} got {got[tuple(bad)]}\n"
                             f"expected:\n{want}\ngot:\n{got}")


async def run_block(dut, qz, n, causal, qi, trace, expect_out, check_cycles=True):
    nb = n // 4
    kjs = list(range(qi + 1)) if causal else list(range(nb))
    entries = [t for t in trace if t["qi"] == qi]
    assert [t["kj"] for t in entries] == kjs

    dut.cfg_scale.value = qz.cfg_scale
    await next_cycle(dut)
    dut.blk_clear.value = 1
    for r in range(4):
        dut.q_load_valid.value = 1
        dut.q_load_idx.value = r
        dut.q_load_row.value = pack(qz.q[qi * 4 + r], 8)
        await next_cycle(dut)
        dut.blk_clear.value = 0
    dut.q_load_valid.value = 0

    for ti, kj in enumerate(kjs):
        kt = qz.k[kj * 4:(kj + 1) * 4]          # K tile rows
        vt = qz.v[kj * 4:(kj + 1) * 4]
        for r in range(4):
            dut.kv_load_valid.value = 1
            dut.kv_load_idx.value = r
            dut.kt_row.value = pack(kt[:, r], 8)   # row r of K^T = column r of K
            dut.v_row.value = pack(vt[r], 8)
            await next_cycle(dut)
        dut.kv_load_valid.value = 0
        last = ti == len(kjs) - 1
        dut.tile_start.value = 1
        dut.tile_diag.value = int(causal and kj == qi)
        dut.tile_last.value = int(last)
        await next_cycle(dut)
        dut.tile_start.value = 0
        dut.tile_diag.value = 0
        dut.tile_last.value = 0
        cycles = 1  # the start cycle itself has elapsed
        while True:
            await sample(dut)
            if int(dut.tile_done.value):
                break
            await next_cycle(dut)
            cycles += 1
            assert cycles < 200, "tile never finished"
        if check_cycles:
            assert cycles == TILE_CYCLES, f"tile took {cycles} cycles, pinned {TILE_CYCLES}"
        t, ctx = entries[ti], f"N={n} causal={causal} qi={qi} kj={kj}"
        diff("S", tile_matrix(int(dut.dbg_s_tile.value), 17, 4), t["S"], ctx)
        diff("Z", tile_matrix(int(dut.dbg_z_tile.value), 16, 4), t["Z"], ctx)
        p_flat = unpack(int(dut.dbg_p_tile.value), 9, 16, signed=False)  # P is unsigned
        diff("P", [p_flat[r * 4:(r + 1) * 4] for r in range(4)], t["P"], ctx)
        diff("alpha", unpack(int(dut.dbg_alpha.value), 16, 4, signed=False), t["alpha"], ctx)
        diff("m", unpack(int(dut.dbg_m.value), 16, 4), t["m"], ctx)
        diff("l", unpack(int(dut.dbg_l.value), 18, 4, signed=False), t["l"], ctx)
        diff("O", tile_matrix(int(dut.dbg_o.value), 26, 4), t["O"], ctx)
        if not last:
            await next_cycle(dut)

    # normalized output rows follow the last tile_done
    got = {}
    for _ in range(12):
        await next_cycle(dut)
        await sample(dut)
        if int(dut.out_valid.value):
            got[int(dut.out_idx.value)] = unpack(int(dut.out_row.value), 16, 4)
            if int(dut.blk_done.value):
                break
    assert sorted(got) == [0, 1, 2, 3], f"output rows {sorted(got)}"
    diff("out", [got[i] for i in range(4)], expect_out[qi * 4:(qi + 1) * 4], f"N={n} causal={causal} qi={qi} (normalized)")
    await next_cycle(dut)
    await sample(dut)
    assert int(dut.tile_busy.value) == 0


def make_inputs(rng, n, amp):
    q = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    k = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    v = np.array([[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)])
    return q, k, v


async def scenario(dut, qz, n, causal, qi):
    res = fixed.flash(qz, n, causal, trace=True)
    await run_block(dut, qz, n, causal, qi, res.trace, res.out)


@cocotb.test()
async def random_blocks(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(31)
    for trial in range(36):
        n = rng.choice([4, 8, 16, 32])
        causal = rng.random() < 0.5
        qi = rng.randrange(n // 4)
        qz = fixed.quantize(*make_inputs(rng, n, rng.choice([0.5, 1.5, 2.5, 3.5])))
        await scenario(dut, qz, n, causal, qi)


@cocotb.test()
async def corner_cases(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(32)
    n = 16
    z0 = np.zeros((n, 4))
    ones = np.ones((n, 4))
    cases = {
        "uniform_maxV": (z0, z0, 127 * ones),          # all logits equal, V at +max: l and O at their maxima
        "uniform_minV": (z0, z0, -127 * ones),
        "max_S": (127 * ones, 127 * ones, 127 * ones),  # |S| = 64516
    }
    for name, (q, k, v) in cases.items():
        qz = fixed.quantize(q, k, v)
        for causal in (False, True):
            await scenario(dut, qz, n, causal, n // 4 - 1)
    # saturating Z: huge cfg_scale pushes logits to +-32767
    q, k, v = make_inputs(rng, n, 2.0)
    qz = fixed.quantize(q, k, v)
    qz.cfg_scale = (1 << 24) - 1
    for causal in (False, True):
        await scenario(dut, qz, n, causal, 2)
    # tiny cfg_scale: every logit rounds to ~0 -> uniform attention
    qz = fixed.quantize(q, k, v)
    qz.cfg_scale = 3
    await scenario(dut, qz, n, False, 1)
    # one dominant key per row: P collapses to a one-hot
    q2, k2, v2 = make_inputs(rng, n, 0.2)
    q2[:, 0], k2[3, 0] = 6.0, 6.0
    await scenario(dut, fixed.quantize(q2, k2, v2), n, False, 0)
    # first-row-only causal: row 0 of block 0 attends to a single key
    await scenario(dut, fixed.quantize(*make_inputs(rng, 4, 2.0)), 4, True, 0)


@cocotb.test()
async def blocks_back_to_back(dut):
    """blk_clear must wipe m, l and O between Q blocks of the same run."""
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(33)
    n = 16
    qz = fixed.quantize(*make_inputs(rng, n, 2.0))
    res = fixed.flash(qz, n, False, trace=True)
    for qi in (0, 3, 1, 2, 0):
        await run_block(dut, qz, n, False, qi, res.trace, res.out)
