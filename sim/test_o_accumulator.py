"""o_accumulator: O = (O*alpha)>>>15 + (pv<<<4), vs fixed.o_update. Operands stay inside the proven 26-bit range."""

import random

import cocotb
import numpy as np
from cocotb.triggers import Timer

from tbutil import fixed, pack, unpack


@cocotb.test()
async def random_rows(dut):
    rng = random.Random(22)
    lim_o, lim_pv = (1 << 24), 130048
    cases = [([0] * 4, 32768, [0] * 4), ([-lim_o] * 4, 32768, [lim_pv] * 4), ([lim_o] * 4, 0, [-lim_pv] * 4),
             ([-1] * 4, 1, [0] * 4), ([-1] * 4, 32768, [0] * 4), ([1] * 4, 16384, [-1] * 4)]
    for _ in range(40000):
        cases.append(([rng.randint(-lim_o, lim_o) for _ in range(4)], rng.choice([0, 32768, rng.randint(0, 32768)]),
                      [rng.randint(-lim_pv, lim_pv) for _ in range(4)]))
    for o, alpha, pv in cases:
        dut.o_old.value = pack(o, 26)
        dut.alpha.value = alpha
        dut.pv.value = pack(pv, 18)
        await Timer(1, "ns")
        got = unpack(int(dut.o_new.value), 26, 4)
        want = fixed.o_update(np.array([o]), np.array([alpha]), np.array([pv]))[0].tolist()
        assert got == want, (o, alpha, pv, got, want)
