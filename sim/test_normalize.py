"""normalize: (o*r + 2^(7+k)) >>> (8+k), clipped to int16, vs fixed.normalize. Includes saturation."""

import random

import cocotb
import numpy as np
from cocotb.triggers import Timer

from tbutil import fixed, to_signed


@cocotb.test()
async def random_elements(dut):
    rng = random.Random(23)
    cases = [(0, 1), (1, 1), (-1, 1), ((1 << 25) - 1, 1), (-(1 << 25), 1), ((1 << 25) - 1, 1 << 17), (1 << 24, 3)]
    for _ in range(40000):
        l = rng.randint(1, (1 << rng.randint(1, 18)) - 1)
        mag = rng.choice([1 << 25, 1 << 20, 1 << 14, 1 << 8])
        cases.append((rng.randint(-mag, mag - 1), l))
    for o, l in cases:
        r, k = fixed.recip(l)
        dut.o.value = o & ((1 << 26) - 1)
        dut.r.value = r
        dut.k.value = k
        await Timer(1, "ns")
        got = to_signed(int(dut.out.value), 16)
        want = int(fixed.normalize(np.array([o]), l)[0])
        assert got == want, (o, l, got, want)
