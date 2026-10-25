"""DMA burst engine against the HBM model (harness: sim/harness/dma_tb.sv).

Timing is pinned exactly: a read burst of W words is done in cycle L + W + 2 (cycle 0 = the cycle start is high),
a posted write is done in cycle W + 1. Counters count 4 bytes per word and are cleared by start.
"""

import random

import cocotb

from tbutil import next_cycle, reset, sample, start_clock


async def idle(dut):
    for sig in ("start", "dir", "ld_en", "stg_we"):
        getattr(dut, sig).value = 0
    dut.hbm_addr.value = 0
    dut.len.value = 0
    dut.cfg_latency.value = 0
    dut.ld_addr.value = 0
    dut.ld_word.value = 0
    dut.host_rd_addr.value = 0
    dut.stg_waddr.value = 0
    dut.stg_wdata.value = 0


async def host_load(dut, addr, words):
    for i, w in enumerate(words):
        dut.ld_en.value = 1
        dut.ld_addr.value = addr + i
        dut.ld_word.value = w
        await next_cycle(dut)
    dut.ld_en.value = 0


async def burst(dut, latency, direction, addr, n):
    """Start a burst and run it to done. Returns (cycle at which done was seen, words seen as {index: data})."""
    dut.cfg_latency.value = latency
    dut.dir.value = direction
    dut.hbm_addr.value = addr
    dut.len.value = n
    dut.start.value = 1
    await next_cycle(dut)  # the edge that captures start: cycle 0 is the start cycle
    dut.start.value = 0
    seen = {}
    k = 1
    while True:
        await sample(dut)
        if int(dut.wd_valid.value):
            seen[int(dut.wd_idx.value)] = int(dut.wd_data.value)
        if int(dut.done.value):
            return k, seen
        await next_cycle(dut)
        k += 1
        assert k < 10_000, "burst never finished"


@cocotb.test()
async def read_bursts(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(41)
    mem = [rng.getrandbits(32) for _ in range(256)]
    await host_load(dut, 0, mem)
    for latency in (0, 1, 3, 17, 64):
        for n in (1, 4, 8, 32, 64):
            addr = rng.randrange(0, 256 - n)
            k, seen = await burst(dut, latency, 0, addr, n)
            assert k == latency + n + 2, f"read L={latency} W={n}: done at {k}, pinned {latency + n + 2}"
            assert [seen[i] for i in range(n)] == mem[addr:addr + n], f"read data L={latency} W={n}"
            await next_cycle(dut)
            await sample(dut)
            assert int(dut.rd_bytes.value) == 4 * n, "rd_bytes must count this burst only"
            assert int(dut.wr_bytes.value) == 0
            await next_cycle(dut)


@cocotb.test()
async def write_bursts(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    rng = random.Random(42)
    for n in (1, 4, 8, 32, 64):
        data = [rng.getrandbits(32) for _ in range(n)]
        for i, w in enumerate(data):  # staging buffer contents
            dut.stg_we.value = 1
            dut.stg_waddr.value = i
            dut.stg_wdata.value = w
            await next_cycle(dut)
        dut.stg_we.value = 0
        addr = rng.randrange(0, 256 - n)
        k, seen = await burst(dut, 9, 1, addr, n)  # latency must not apply to writes
        assert k == n + 1, f"write W={n}: done at {k}, pinned {n + 1}"
        assert not seen, "a write burst must not produce read words"
        await next_cycle(dut)
        for i in range(n):
            dut.host_rd_addr.value = addr + i
            await sample(dut)
            assert int(dut.host_rd_word.value) == data[i], f"write data W={n} word {i}"
            await next_cycle(dut)
        await sample(dut)
        assert int(dut.wr_bytes.value) == 4 * n and int(dut.rd_bytes.value) == 0
        await next_cycle(dut)


@cocotb.test()
async def start_ignored_while_busy(dut):
    await start_clock(dut)
    await idle(dut)
    await reset(dut)
    mem = list(range(100, 140))
    await host_load(dut, 0, mem)
    dut.cfg_latency.value = 2
    dut.dir.value = 0
    dut.hbm_addr.value = 0
    dut.len.value = 8
    dut.start.value = 1
    await next_cycle(dut)
    dut.start.value = 0
    dut.hbm_addr.value = 30  # must be ignored: the burst keeps its original address
    dut.len.value = 1
    got = {}
    for _ in range(40):
        await sample(dut)
        if int(dut.wd_valid.value):
            got[int(dut.wd_idx.value)] = int(dut.wd_data.value)
        if int(dut.done.value):
            break
        await next_cycle(dut)
    dut.start.value = 1
    await next_cycle(dut)  # a start while idle after done is a new burst, not the old one
    dut.start.value = 0
    assert [got[i] for i in range(8)] == mem[0:8]
