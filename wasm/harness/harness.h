// TinyAttnSim: one instance of the tiny_attn_top model with its own VerilatedContext, so several instances can run in
// one WebAssembly module side by side (L4 of the visualizer runs NAIVE and FLASH at once).
//
// The engine is driven exactly as a host would drive it: Q/K/V are written through the HBM load port, start is a
// one-cycle pulse, O is read back through the readback port. Every step() is one rising clock edge.
#pragma once

#include <cstdint>
#include <vector>

#include "Vtiny_attn_top.h"
#include "verilated.h"

struct Snapshot {
    uint32_t cycle;      // clock edges since reset
    uint32_t ctrlState;  // flash_ctrl / naive_ctrl state (dbg_ctrl_state)
    bool busy;
    bool done;
    uint32_t hbmRdBytes;
    uint32_t hbmWrBytes;
    uint32_t cycles;     // busy cycles (perf counter)
    uint32_t tiles;
    uint32_t kvLoads;
};

class TinyAttnSim {
public:
    TinyAttnSim();
    ~TinyAttnSim();
    TinyAttnSim(const TinyAttnSim&) = delete;
    TinyAttnSim& operator=(const TinyAttnSim&) = delete;

    void reset();
    // n: sequence length (multiple of 4, 4..32); qrows: rows per Q block (4..32, <= n); mode: 0 FLASH, 1 NAIVE.
    void configure(int n, int qrows, bool causal, int mode, bool overlap, int latency, uint32_t scale);
    // q, k, v: n*4 int8 values, row-major. Written through the HBM load port (takes n*3 cycles).
    void loadQKV(const std::vector<int32_t>& q, const std::vector<int32_t>& k, const std::vector<int32_t>& v);
    // Starts the engine (one-cycle start pulse). The next step() is the first cycle of the run.
    void start();
    // One clock. Returns the state after the edge.
    Snapshot step();
    // Steps until done or maxCycles edges. Returns the number of edges taken, or -1 on timeout.
    int runToDone(int maxCycles);
    // O for the configured n: n*4 int16 values (sign-extended), row-major.
    std::vector<int32_t> readOutput();
    // Raw HBM words [addr, addr+len).
    std::vector<uint32_t> readHbm(int addr, int len);
    Snapshot snapshot() const;
    bool isDone() const;

private:
    void clockEdge();
    void evalComb();

    VerilatedContext* ctx_;
    Vtiny_attn_top* top_;
    int n_ = 4;
    int qrows_ = 4;
    uint32_t cycle_ = 0;
};
