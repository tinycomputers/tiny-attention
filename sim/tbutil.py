"""Shared cocotb helpers: golden-model import path, bit packing, clocking.

Cycle convention used by every sequential test: inputs for "cycle t" are driven 1 ns after rising edge t, and
outputs for cycle t are sampled at the falling edge in the middle of it (registers settled, inputs stable).
So a signal registered once appears one cycle after the input that caused it.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent / "golden"))

import fixed  # noqa: E402,F401  (re-exported for tests)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def to_signed(v: int, w: int) -> int:
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def pack(values, w: int) -> int:
    """values[i] -> bits [i*w +: w] (two's complement)."""
    out = 0
    for i, v in enumerate(values):
        out |= (int(v) & ((1 << w) - 1)) << (i * w)
    return out


def unpack(x: int, w: int, n: int, signed: bool = True):
    vals = [(x >> (i * w)) & ((1 << w) - 1) for i in range(n)]
    return [to_signed(v, w) for v in vals] if signed else vals


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())


async def reset(dut, cycles: int = 2):
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await Timer(1, "ns")
    dut.rst_n.value = 1


async def next_cycle(dut):
    """Advance to the start of the next cycle (1 ns after rising edge): safe point to drive inputs."""
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


async def sample(dut):
    """Wait to mid-cycle and return; read outputs after this."""
    await FallingEdge(dut.clk)
