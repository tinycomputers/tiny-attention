import random

import cocotb
from cocotb.triggers import Timer

from tbutil import pack


@cocotb.test()
async def max_of_four(dut):
    rng = random.Random(3)
    cases = [[-32768] * 4, [32767] * 4, [-32768, -32767, 0, -1], [5, 5, 5, 5], [-32768, -32768, 7, -32768]]
    for _ in range(5000):
        cases.append([rng.choice([-32768, rng.randint(-32767, 32767)]) for _ in range(4)])
    for vals in cases:
        dut.x.value = pack(vals, 16)
        await Timer(1, "ns")
        got = int(dut.max_o.value)
        got = got - 65536 if got >= 32768 else got
        assert got == max(vals), (vals, got)
