import random

import cocotb
import numpy as np
from cocotb.triggers import Timer

from tbutil import fixed, to_signed


@cocotb.test()
async def matches_golden(dut):
    rng = random.Random(5)
    cases = [(0, 0, 1), (64516, (1 << 24) - 1, 1), (-64516, (1 << 24) - 1, 1), (64516, 1 << 20, 1),
             (-64516, 1 << 20, 1), (1, 1, 1), (-1, 1, 1), (12345, 756388, 0), (-5, 3, 0), (127, 1 << 12, 1)]
    for _ in range(30000):
        s = rng.randint(-64516, 64516)
        scale = rng.choice([rng.randint(0, (1 << 24) - 1), rng.randint(0, 1 << 14), rng.randint(1 << 18, 1 << 21)])
        cases.append((s, scale, rng.randint(0, 1)))
    for s, scale, valid in cases:
        dut.s.value = s & 0x1FFFF
        dut.scale.value = scale
        dut.valid.value = valid
        await Timer(1, "ns")
        got = to_signed(int(dut.z.value), 16)
        want = int(fixed.scale_mask(np.array([s]), scale, np.array([bool(valid)]))[0])
        assert got == want, (s, scale, valid, got, want)
