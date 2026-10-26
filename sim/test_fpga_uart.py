"""The FPGA top (fpga/ecp5/fpga_top.sv) driven over its UART, the way a host would drive the board.

A host model sends commands on uart_rx and reads replies on uart_tx at CLKS_PER_BIT clocks per bit (set with
EXTRA_ARGS="-GCLKS_PER_BIT=4" in the run script; the board uses 217 for 25 MHz and 115200 baud). Checked:
  * the O words read back over UART equal the golden model, bit for bit
  * the DONE reply's cycle, HBM read and write byte counts equal the closed-form models (FLASH: schedule.py, traffic.py)
"""

import random

import cocotb
from cocotb.triggers import RisingEdge, Timer

from tbutil import fixed, pack, reset, start_clock, unpack  # sets up the golden-model import path
from schedule import cycles as sched_cycles
from traffic import traffic

Q_BASE, K_BASE, V_BASE, O_BASE = 0, 32, 64, 96
CMD_CONFIG, CMD_LOAD, CMD_START, CMD_READ = 0x01, 0x02, 0x03, 0x04
REP_DONE, REP_WORD = 0x83, 0x84


class Host:
    """UART host: bit-banged at CPB clocks per bit, LSB first, 8N1."""

    def __init__(self, dut, cpb: int):
        self.dut = dut
        self.cpb = cpb

    async def _hold(self, level: int, cycles: int):
        for _ in range(cycles):
            self.dut.uart_rx.value = level
            await RisingEdge(self.dut.clk)
            await Timer(1, "ns")

    async def send(self, data: bytes):
        for b in data:
            await self._hold(0, self.cpb)  # start
            for i in range(8):
                await self._hold((b >> i) & 1, self.cpb)
            await self._hold(1, self.cpb)  # stop

    async def recv(self, n: int, timeout: int = 2_000_000) -> bytes:
        """n bytes from uart_tx: find each start edge, sample every data bit at its middle."""
        out = bytearray()
        for _ in range(n):
            waited = 0
            while int(self.dut.uart_tx.value) == 1:
                await self._advance(1)
                waited += 1
                if waited > timeout:
                    raise AssertionError("no reply from the device")
            # now at the first cycle of the start bit; data bit k is centred at (k + 1.5) bit periods
            await self._advance(self.cpb + self.cpb // 2)
            byte = 0
            for k in range(8):
                byte |= int(self.dut.uart_tx.value) << k
                await self._advance(self.cpb)
            out.append(byte)  # we are now in the middle of the stop bit
        return bytes(out)

    async def _advance(self, cycles: int):
        for _ in range(cycles):
            await RisingEdge(self.dut.clk)
            await Timer(1, "ns")

    async def config(self, nb, gg, causal, naive, overlap, latency, scale):
        flags = (1 if causal else 0) | (2 if naive else 0) | (4 if overlap else 0)
        await self.send(bytes([CMD_CONFIG, nb, gg, flags, latency]) + scale.to_bytes(3, "little"))

    async def load(self, addr, word):
        await self.send(bytes([CMD_LOAD]) + addr.to_bytes(2, "little") + (word & 0xFFFFFFFF).to_bytes(4, "little"))

    async def start(self):
        await self.send(bytes([CMD_START]))
        rep = await self.recv(13)
        assert rep[0] == REP_DONE, f"bad reply code {rep[0]:#x}"
        vals = [int.from_bytes(rep[1 + 4 * i: 5 + 4 * i], "little") for i in range(3)]
        return dict(cycles=vals[0], rd=vals[1], wr=vals[2])

    async def read(self, addr):
        await self.send(bytes([CMD_READ]) + addr.to_bytes(2, "little"))
        rep = await self.recv(5)
        assert rep[0] == REP_WORD, f"bad reply code {rep[0]:#x}"
        return int.from_bytes(rep[1:5], "little")


async def run_case(dut, cpb, rng, n, causal, naive, qrows):
    q = [[rng.gauss(0, 2.5) for _ in range(4)] for _ in range(n)]
    k = [[rng.gauss(0, 2.5) for _ in range(4)] for _ in range(n)]
    v = [[rng.gauss(0, 1.0) for _ in range(4)] for _ in range(n)]
    qz = fixed.quantize(q, k, v)

    host = Host(dut, cpb)
    await host.config(n // 4, (4 if naive else qrows) // 4, causal, naive, False, 10, qz.cfg_scale)
    for i in range(n):
        await host.load(Q_BASE + i, pack(qz.q[i], 8))
    for i in range(n):
        await host.load(K_BASE + i, pack(qz.k[i], 8))
    for i in range(n):
        await host.load(V_BASE + i, pack(qz.v[i], 8))

    rep = await host.start()
    qblock = 4 if naive else qrows
    t = traffic(n, causal, "NAIVE" if naive else "FLASH", qblock=qblock)
    ctx = f"{'NAIVE' if naive else 'FLASH'} N={n} causal={causal} qrows={qblock}"
    assert rep["rd"] == t["rd"], f"{ctx}: read bytes {rep['rd']} != {t['rd']}"
    assert rep["wr"] == t["wr"], f"{ctx}: write bytes {rep['wr']} != {t['wr']}"
    if not naive:
        assert rep["cycles"] == sched_cycles(n, causal, qblock, 10), f"{ctx}: cycles {rep['cycles']}"

    got = []
    for i in range(n):
        lo = await host.read(O_BASE + 2 * i)
        hi = await host.read(O_BASE + 2 * i + 1)
        got.append(unpack(lo, 16, 2) + unpack(hi, 16, 2))
    want = (fixed.naive(qz, n, causal) if naive else fixed.flash(qz, n, causal, qblock=qblock)).out.tolist()
    assert got == want, f"{ctx}: O over UART differs from the golden model"


@cocotb.test()
async def uart_host_flow(dut):
    """Configure, load, run and read back over the UART: FLASH and NAIVE, causal and not."""
    import os

    cpb = int(os.environ.get("CLKS_PER_BIT", "4"))
    await start_clock(dut)
    dut.uart_rx.value = 1
    await reset(dut)
    rng = random.Random(71)
    cases = [
        dict(n=8, causal=False, naive=False, qrows=4),
        dict(n=8, causal=True, naive=False, qrows=8),
        dict(n=8, causal=True, naive=True, qrows=4),
    ]
    for c in cases:
        await run_case(dut, cpb, rng, **c)
