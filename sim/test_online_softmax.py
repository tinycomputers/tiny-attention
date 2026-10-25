"""online_softmax: one row step vs fixed.online_softmax_step, incl. masks, running max rising/falling/equal."""

import random

import cocotb
import numpy as np
from cocotb.triggers import Timer

from tbutil import fixed, pack, to_signed

MASK = -32768


def unpack_p(x):
    return [(x >> (j * 9)) & 0x1FF for j in range(4)]


async def check(dut, z, m, l):
    dut.z.value = pack(z, 16)
    dut.m_old.value = m & 0xFFFF
    dut.l_old.value = l
    await Timer(1, "ns")
    m_new, alpha, p, l_new = fixed.online_softmax_step(np.array([z]), np.array([m]), np.array([l]))
    got = (to_signed(int(dut.m_new.value), 16), int(dut.alpha.value), unpack_p(int(dut.p.value)), int(dut.l_new.value))
    want = (int(m_new[0]), int(alpha[0]), p[0].tolist(), int(l_new[0]))
    assert got == want, (z, m, l, got, want)


@cocotb.test()
async def random_rows(dut):
    rng = random.Random(21)
    await check(dut, [0, 0, 0, 0], MASK, 0)  # first tile of a row: m = init, l = 0
    await check(dut, [5, MASK, MASK, MASK], MASK, 0)  # only one unmasked element
    await check(dut, [32767, -32767, 32767, 0], MASK, 0)  # saturated logits
    await check(dut, [100, 200, 300, 400], 400, 5000)  # running max equal to new max: alpha = 1.0
    for _ in range(40000):
        spread = rng.choice([50, 400, 3000, 32767])
        z = [rng.randint(-spread, spread) for _ in range(4)]
        for j in range(4):
            if rng.random() < 0.15:
                z[j] = MASK
        if all(v == MASK for v in z):
            z[rng.randrange(4)] = rng.randint(-spread, spread)
        m = rng.choice([MASK, rng.randint(-spread, spread), max(z) + rng.randint(-5, 5)])
        m = max(min(m, 32767), -32768)
        l = rng.choice([0, rng.randint(0, 1 << 17), rng.randint(0, 5000)])
        await check(dut, z, m, l)
