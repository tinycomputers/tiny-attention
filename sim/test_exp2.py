"""exp2_unit: EXHAUSTIVE over every Q9.8 input <= 0 (65 537 values) against fixed.exp2_q."""

import cocotb
import numpy as np
from cocotb.triggers import Timer

from tbutil import fixed


@cocotb.test()
async def exhaustive(dut):
    xs = np.arange(-65536, 1)
    want = fixed.exp2_q(xs)
    for x, w in zip(xs.tolist(), want.tolist()):
        dut.x.value = x & 0x1FFFF
        await Timer(1, "ns")
        got = int(dut.y.value)
        assert got == w, (x, got, w)
    # outside the contract: x > 0 saturates
    for x in (1, 255, 256, 65535):
        dut.x.value = x
        await Timer(1, "ns")
        assert int(dut.y.value) == 0xFFFF, x
