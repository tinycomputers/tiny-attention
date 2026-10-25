"""attn_tile lockstep: Q blocks (up to 8 row groups) against their K/V tiles, compared with fixed.flash(trace=True) at
EVERY tile boundary (S, Z, P, alpha, m, l, O) and on the normalized rows of each block. The test drives the tile core
the way flash_ctrl does: one kv_load per K/V tile, then one tile_start per unmasked row group.
"""

import random

import cocotb
import numpy as np

from tbutil import fixed, next_cycle, pack, reset, sample, start_clock, unpack

TILE_CYCLES = 20  # tile_start -> tile_done, see rtl/attn_tile.sv


async def idle(dut):
    for sig in ("blk_clear", "q_load_valid", "kv_load_valid", "tile_start", "tile_diag", "blk_norm"):
        getattr(dut, sig).value = 0
    for sig in ("q_load_idx", "q_load_row", "kv_load_idx", "kt_row", "v_row", "tile_grp", "blk_nrows"):
        getattr(dut, sig).value = 0
    dut.cfg_scale.value = 0


def diff(name, got, want, ctx):
    got, want = np.array(got), np.array(want)
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)[0]
        raise AssertionError(
            f"{ctx}: {name} differs at {tuple(int(b) for b in bad)}: expected {want[tuple(bad)]} got {got[tuple(bad)]}\n"
            f"expected:\n{want}\ngot:\n{got}"
        )


def mat(x, w, rows, cols, signed=True):
    flat = unpack(x, w, rows * cols, signed)
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


async def run_block(dut, qz, n, causal, gbase, bg, trace, expect_out):
    """One Q block: rows gbase*4 .. gbase*4 + 4*bg - 1 of the sequence."""
    nb = n // 4
    last = gbase + bg - 1 if causal else nb - 1
    dut.cfg_scale.value = qz.cfg_scale
    dut.blk_clear.value = 1
    await next_cycle(dut)
    dut.blk_clear.value = 0
    for r in range(4 * bg):
        dut.q_load_valid.value = 1
        dut.q_load_idx.value = r
        dut.q_load_row.value = pack(qz.q[gbase * 4 + r], 8)
        await next_cycle(dut)
    dut.q_load_valid.value = 0

    for kj in range(last + 1):
        kt = qz.k[kj * 4:(kj + 1) * 4]
        vt = qz.v[kj * 4:(kj + 1) * 4]
        for r in range(4):
            dut.kv_load_valid.value = 1
            dut.kv_load_idx.value = r
            dut.kt_row.value = pack(kt[:, r], 8)  # row r of Kᵀ = column r of the K tile
            dut.v_row.value = pack(vt[r], 8)
            await next_cycle(dut)
        dut.kv_load_valid.value = 0

        for gi in range(bg):
            g = gbase + gi
            if causal and kj > g:
                continue
            dut.tile_start.value = 1
            dut.tile_grp.value = gi
            dut.tile_diag.value = int(causal and kj == g)
            await next_cycle(dut)
            dut.tile_start.value = 0
            dut.tile_diag.value = 0
            cycles = 1
            while True:
                await sample(dut)
                if int(dut.tile_done.value):
                    break
                await next_cycle(dut)
                cycles += 1
                assert cycles < 200, "tile never finished"
            assert cycles == TILE_CYCLES, f"tile took {cycles} cycles, pinned {TILE_CYCLES}"
            assert int(dut.dbg_cur_grp.value) == gi

            t = trace[(g, kj)]
            ctx = f"N={n} causal={causal} group={g} kj={kj}"
            diff("S", mat(int(dut.dbg_s_tile.value), 17, 4, 4), t["S"], ctx)
            diff("Z", mat(int(dut.dbg_z_tile.value), 16, 4, 4), t["Z"], ctx)
            diff("P", mat(int(dut.dbg_p_tile.value), 9, 4, 4, signed=False), t["P"], ctx)
            diff("alpha", unpack(int(dut.dbg_alpha.value), 16, 4, signed=False), t["alpha"], ctx)
            diff("m", unpack(int(dut.dbg_m.value), 16, 4), t["m"], ctx)
            diff("l", unpack(int(dut.dbg_l.value), 18, 4, signed=False), t["l"], ctx)
            diff("O", mat(int(dut.dbg_o.value), 26, 4, 4), t["O"], ctx)
            await next_cycle(dut)

    dut.blk_nrows.value = 4 * bg
    dut.blk_norm.value = 1
    await next_cycle(dut)
    dut.blk_norm.value = 0
    got = {}
    for _ in range(8 * bg + 8):
        await sample(dut)
        if int(dut.out_valid.value):
            got[int(dut.out_idx.value)] = unpack(int(dut.out_row.value), 16, 4)
        if int(dut.blk_done.value):
            break
        await next_cycle(dut)
    assert sorted(got) == list(range(4 * bg)), f"output rows {sorted(got)}"
    rows = [got[i] for i in range(4 * bg)]
    diff("out", rows, expect_out[gbase * 4:gbase * 4 + 4 * bg], f"N={n} causal={causal} block@{gbase} (normalized)")
    await next_cycle(dut)
    await sample(dut)
    assert int(dut.tile_busy.value) == 0
    await next_cycle(dut)


def make_inputs(rng, n, amp):
    q = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    k = np.array([[rng.gauss(0, amp) for _ in range(4)] for _ in range(n)])
    v = np.array([[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)])
    return q, k, v


async def run_sequence(dut, qz, n, causal, qrows, blocks):
    """Run the given blocks (group bases) of one sequence against the golden trace."""
    res = fixed.flash(qz, n, causal, trace=True)
    trace = {(t["qi"], t["kj"]): t for t in res.trace}
    G = qrows // 4
    for gbase in blocks:
        bg = min(G, n // 4 - gbase)
        await run_block(dut, qz, n, causal, gbase, bg, trace, res.out)


@cocotb.test()
async def random_blocks(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(31)
    for _ in range(40):
        n = rng.choice([4, 8, 12, 16, 20, 24, 28, 32])
        causal = rng.random() < 0.5
        qrows = rng.choice([q for q in (4, 8, 16, 32) if q <= n])
        qz = fixed.quantize(*make_inputs(rng, n, rng.choice([0.5, 1.5, 2.5, 3.5])))
        G, nb = qrows // 4, n // 4
        blocks = list(range(0, nb, G))
        await run_sequence(dut, qz, n, causal, qrows, blocks)


@cocotb.test()
async def corner_cases(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(32)
    n = 32
    z0 = np.zeros((n, 4))
    ones = np.ones((n, 4))
    cases = {
        "uniform_maxV": (z0, z0, 127 * ones),  # equal logits, V at +max: l and O at their maxima
        "uniform_minV": (z0, z0, -127 * ones),
        "max_S": (127 * ones, 127 * ones, 127 * ones),  # |S| = 4*127*127
    }
    for name, (q, k, v) in cases.items():
        for causal in (False, True):
            await run_sequence(dut, fixed.quantize(q, k, v), n, causal, 32, [0])
    # saturating Z: huge cfg_scale pushes logits to +-32767
    qz = fixed.quantize(*make_inputs(rng, n, 2.0))
    qz.cfg_scale = (1 << 24) - 1
    for causal in (False, True):
        await run_sequence(dut, qz, n, causal, 32, [0])
    # near-zero cfg_scale: uniform attention
    qz = fixed.quantize(*make_inputs(rng, n, 2.0))
    qz.cfg_scale = 3
    await run_sequence(dut, qz, n, False, 32, [0])
    # one dominant key: P collapses to one-hot
    q2, k2, v2 = make_inputs(rng, n, 0.2)
    q2[:, 0], k2[17, 0] = 6.0, 6.0
    await run_sequence(dut, fixed.quantize(q2, k2, v2), n, False, 32, [0])


@cocotb.test()
async def blocks_back_to_back(dut):
    """blk_clear must wipe m, l and O between blocks of the same run (each block re-clears)."""
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(33)
    n = 16
    qz = fixed.quantize(*make_inputs(rng, n, 2.0))
    for gbase in (0, 2, 1, 3, 0):
        res = fixed.flash(qz, n, False, trace=True)
        trace = {(t["qi"], t["kj"]): t for t in res.trace}
        await run_block(dut, qz, n, False, gbase, 1, trace, res.out)
