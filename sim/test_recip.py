"""recip_unit: exhaustive over all 32 768 mantissas (and their low-bit truncation), every exponent boundary, and a
log-uniform random sweep, all against fixed.recip."""

import random

import cocotb
from cocotb.triggers import Timer

from tbutil import fixed


async def check(dut, l):
    dut.l.value = l
    await Timer(1, "ns")
    r, k = fixed.recip(l)
    assert (int(dut.r.value), int(dut.k.value)) == (r, k), (l, int(dut.r.value), int(dut.k.value), r, k)


@cocotb.test()
async def every_mantissa(dut):
    for m16 in range(1 << 15, 1 << 16):
        await check(dut, m16 << 2)  # k = 17, low bits zero
    rng = random.Random(9)
    for m16 in range(1 << 15, 1 << 16, 7):
        await check(dut, (m16 << 2) | rng.randint(0, 3))  # low bits are truncated by the normalizer


@cocotb.test()
async def every_exponent(dut):
    for k in range(18):
        for l in {1 << k, (1 << (k + 1)) - 1, (1 << k) | 1, (1 << k) + (1 << k) // 2}:
            if l >= 1:
                await check(dut, l)


@cocotb.test()
async def random_sweep(dut):
    rng = random.Random(10)
    for _ in range(30000):
        await check(dut, rng.randint(1, (1 << rng.randint(1, 18)) - 1))


@cocotb.test()
async def zero_is_defined(dut):
    dut.l.value = 0
    await Timer(1, "ns")
    assert int(dut.r.value) == 0 and int(dut.k.value) == 0
