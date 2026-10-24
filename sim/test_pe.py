"""PE: weight load, MAC, activation pass-through (one register each)."""

import random

import cocotb
from cocotb.triggers import Timer

from tbutil import env_int, next_cycle, reset, sample, start_clock, to_signed

A_W = env_int("A_W", 8)
A_SIGNED = env_int("A_SIGNED", 1)
ACC_W = env_int("ACC_W", 17)


def a_val(v):  # DUT sees the raw bits
    return v & ((1 << A_W) - 1)


def a_range():
    return (-(1 << (A_W - 1)) + 1, (1 << (A_W - 1)) - 1) if A_SIGNED else (0, 1 << (A_W - 1))


@cocotb.test()
async def weight_load_and_mac(dut):
    await start_clock(dut)
    dut.w_we.value = 0
    dut.w_in.value = 0
    dut.a_in.value = 0
    dut.psum_in.value = 0
    await reset(dut)
    rng = random.Random(1)
    lo, hi = a_range()
    w = rng.randint(-127, 127)
    await next_cycle(dut)
    dut.w_we.value = 1
    dut.w_in.value = w & 0xFF
    await next_cycle(dut)
    dut.w_we.value = 0
    dut.w_in.value = 0x55  # must be ignored without w_we
    await sample(dut)
    assert to_signed(int(dut.dbg_w.value), 8) == w
    for _ in range(500):
        a, ps = rng.randint(lo, hi), rng.randint(-20000, 20000)
        dut.a_in.value = a_val(a)
        dut.psum_in.value = ps & ((1 << ACC_W) - 1)
        await next_cycle(dut)  # the edge that captured them
        await sample(dut)
        assert to_signed(int(dut.psum_out.value), ACC_W) == ps + a * w, (a, w, ps)
        assert int(dut.a_out.value) == a_val(a)


@cocotb.test()
async def weight_holds_and_reset_clears(dut):
    await start_clock(dut)
    dut.w_we.value = 1
    dut.w_in.value = 0x2A
    dut.a_in.value = 3
    dut.psum_in.value = 0
    await reset(dut)
    await next_cycle(dut)
    await next_cycle(dut)
    await sample(dut)
    assert to_signed(int(dut.dbg_w.value), 8) == 0x2A
    dut.rst_n.value = 0
    await Timer(1, "ns")
    assert int(dut.dbg_w.value) == 0 and int(dut.psum_out.value) == 0
