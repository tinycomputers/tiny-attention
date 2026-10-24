"""Phase 0 spike: proves cocotb + Verilator work end to end."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.en.value = 0
    dut.rst_n.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def ticks(dut, n):
    for _ in range(n):
        await RisingEdge(dut.clk)
    await ReadOnly()
    return int(dut.debug_bus.value)


@cocotb.test()
async def reset_clears(dut):
    await start(dut)
    await ReadOnly()
    assert int(dut.debug_bus.value) == 0


@cocotb.test()
async def counts_when_enabled(dut):
    await start(dut)
    dut.en.value = 1
    assert await ticks(dut, 5) == 5


@cocotb.test()
async def holds_when_disabled(dut):
    await start(dut)
    dut.en.value = 1
    assert await ticks(dut, 3) == 3
    from cocotb.triggers import Timer

    await Timer(1, "ns")
    dut.en.value = 0
    assert await ticks(dut, 4) == 3


@cocotb.test()
async def wraps_at_256(dut):
    await start(dut)
    dut.en.value = 1
    assert await ticks(dut, 256) == 0
