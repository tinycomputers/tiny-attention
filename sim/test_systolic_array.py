"""Systolic array with hand-skewed inputs, independent of mm_engine's skew logic.

Row i of A is presented at cycle t_i; element r reaches the array input at t_i + r; column c of the bottom edge
holds C[i][c] during cycle t_i + N + c.
"""

import random

import cocotb

from tbutil import env_int, next_cycle, pack, reset, sample, start_clock, unpack

N = 4
A_W = env_int("A_W", 8)
A_SIGNED = env_int("A_SIGNED", 1)
ACC_W = env_int("ACC_W", 17)


def a_range():
    return (-127, 127) if A_SIGNED else (0, 256)


async def load_weights(dut, w):
    for r in range(N):
        dut.w_we_row.value = 1 << r
        dut.w_data.value = pack(w[r], 8)
        await next_cycle(dut)
    dut.w_we_row.value = 0


@cocotb.test()
async def matmul_wavefront(dut):
    await start_clock(dut)
    dut.w_we_row.value = 0
    dut.w_data.value = 0
    dut.a_in.value = 0
    await reset(dut)
    rng = random.Random(7)
    lo, hi = a_range()
    for trial in range(20):
        w = [[rng.randint(-127, 127) for _ in range(N)] for _ in range(N)]  # W[r][c]
        m = rng.randint(1, 9)
        a = [[rng.randint(lo, hi) for _ in range(N)] for _ in range(m)]
        await next_cycle(dut)
        await load_weights(dut, w)
        await next_cycle(dut)
        expected = [[sum(a[i][r] * w[r][c] for r in range(N)) for c in range(N)] for i in range(m)]
        got = [[None] * N for _ in range(m)]
        start = 0
        total = start + m + 2 * N + 2
        for t in range(start, total):
            # element r of row i enters at cycle t_i + r  ->  at cycle t, row i = t - r for each r
            elems = []
            for r in range(N):
                i = t - r
                elems.append(a[i][r] if 0 <= i < m else 0)
            dut.a_in.value = pack(elems, A_W)
            await sample(dut)
            cols = unpack(int(dut.psum_out.value), ACC_W, N)
            for c in range(N):
                i = t - N - c
                if 0 <= i < m:
                    got[i][c] = cols[c]
            await next_cycle(dut)
        assert got == expected, f"trial {trial}"
        dut.a_in.value = 0
        for _ in range(2 * N):
            await next_cycle(dut)  # flush before reloading weights
