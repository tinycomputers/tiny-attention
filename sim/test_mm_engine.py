"""mm_engine: C = A.W over a streamed A, skew/deskew, valid pipeline, gaps, back-to-back reloads.
Run twice: MM1 (int8 x int8, 17-bit) and MM2 (9-bit unsigned P x int8, 18-bit); see run_unit.sh."""

import random

import cocotb

from tbutil import env_int, next_cycle, pack, reset, sample, start_clock, unpack

N = 4
LATENCY = 2 * N - 1
A_W = env_int("A_W", 8)
A_SIGNED = env_int("A_SIGNED", 1)
ACC_W = env_int("ACC_W", 17)


def a_range():
    return (-127, 127) if A_SIGNED else (0, 256)


async def idle_inputs(dut):
    dut.w_row_valid.value = 0
    dut.w_row_idx.value = 0
    dut.w_row.value = 0
    dut.a_valid.value = 0
    dut.a_row.value = 0


async def load_weights(dut, w):
    for r in range(N):
        dut.w_row_valid.value = 1
        dut.w_row_idx.value = r
        dut.w_row.value = pack(w[r], 8)
        await next_cycle(dut)
    dut.w_row_valid.value = 0


async def run_stream(dut, a, gaps):
    """Present rows of `a` with `gaps[i]` idle cycles before row i. Returns (row_index -> (cycle, values))
    for every c_valid seen, and the cycle each row was presented."""
    presented, results = {}, []
    sched = []
    for i, g in enumerate(gaps):
        sched += [None] * g + [i]
    horizon = len(sched) + LATENCY + 3
    for t in range(horizon):
        item = sched[t] if t < len(sched) else None
        if item is None:
            dut.a_valid.value = 0
            dut.a_row.value = 0xFFFF  # garbage while invalid: must be ignored
        else:
            dut.a_valid.value = 1
            dut.a_row.value = pack(a[item], A_W)
            presented[item] = t
        await sample(dut)
        if int(dut.c_valid.value):
            results.append((t, unpack(int(dut.c_row.value), ACC_W, N)))
        await next_cycle(dut)
    dut.a_valid.value = 0
    return presented, results


@cocotb.test()
async def stream_matches_numpy(dut):
    await start_clock(dut)
    await idle_inputs(dut)
    await reset(dut)
    rng = random.Random(11)
    lo, hi = a_range()
    for trial in range(30):
        w = [[rng.randint(-127, 127) for _ in range(N)] for _ in range(N)]
        m = rng.randint(1, 12)
        a = [[rng.randint(lo, hi) for _ in range(N)] for _ in range(m)]
        gaps = [rng.choice([0, 0, 0, 1, 3]) for _ in range(m)]
        await next_cycle(dut)
        await load_weights(dut, w)
        await next_cycle(dut)
        presented, results = await run_stream(dut, a, gaps)
        assert len(results) == m, f"trial {trial}: {len(results)} outputs for {m} rows"
        for i, (t, vals) in enumerate(results):
            assert t == presented[i] + LATENCY, (trial, i, t, presented[i])
            want = [sum(a[i][r] * w[r][c] for r in range(N)) for c in range(N)]
            assert vals == want, (trial, i, vals, want)
        for _ in range(LATENCY):
            await next_cycle(dut)


@cocotb.test()
async def extremes_fit_accumulator(dut):
    """Max-magnitude operands: the accumulator width in docs/ARCHITECTURE.md is exactly enough."""
    await start_clock(dut)
    await idle_inputs(dut)
    await reset(dut)
    lo, hi = a_range()
    amax = hi if A_SIGNED else 256
    for a_val in {amax, lo}:
        for w_val in (127, -127):
            w = [[w_val] * N for _ in range(N)]
            a = [[a_val] * N for _ in range(N)]
            await next_cycle(dut)
            await load_weights(dut, w)
            await next_cycle(dut)
            _, results = await run_stream(dut, a, [0] * N)
            want = [N * a_val * w_val] * N
            assert all(v == want for _, v in results) and len(results) == N, (a_val, w_val, results[:1], want)
            for _ in range(LATENCY):
                await next_cycle(dut)


@cocotb.test()
async def idle_and_busy(dut):
    await start_clock(dut)
    await idle_inputs(dut)
    await reset(dut)
    for _ in range(20):
        await sample(dut)
        assert int(dut.c_valid.value) == 0 and int(dut.busy.value) == 0
        await next_cycle(dut)
    dut.a_valid.value = 1
    dut.a_row.value = 0
    await sample(dut)
    assert int(dut.busy.value) == 1
    await next_cycle(dut)
    dut.a_valid.value = 0
    seen = 0
    for _ in range(LATENCY + 2):
        await sample(dut)
        seen += int(dut.busy.value)
        await next_cycle(dut)
    assert seen == LATENCY  # busy covers the in-flight cycles, then drops
