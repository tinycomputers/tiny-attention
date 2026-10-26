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
    bool tileDone;       // one-cycle pulse: a tile completed on the last edge
};

// Tile-core state of the most recent tile (see attn_tile.sv dbg_* ports). S, Z, P, m, l, alpha and O are for the
// group of that tile: 4 rows (S/Z/P: 4 x 4 lanes, O: 4 x 4 columns). Values are signed or unsigned as in the RTL.
struct TileState {
    int curGrp;
    int srow;
    std::vector<int32_t> s;      // 16, S_W = 17 bit signed
    std::vector<int32_t> z;      // 16, Q8.8 signed
    std::vector<int32_t> p;      // 16, UQ1.8 (0..256)
    std::vector<int32_t> alpha;  // 4, UQ1.15
    std::vector<int32_t> m;      // 4, Q8.8 signed
    std::vector<int32_t> l;      // 4, 18 bit unsigned
    std::vector<int32_t> o;      // 16, 26 bit signed
};

// K and V tile buffers. k[(buf*4 + row)*4 + col] is K[row][col] of buffer buf (int8, sign-extended); same for v.
// sel is the buffer the engine is using for the current tile.
// Datapath wires of the two systolic arrays, live. Index r*4 + c is PE (r, c). MM1 computes S = Q . Kt (weights Kt,
// activations Q rows); MM2 computes P . V (weights V, activations P). Values are signed except mm2A (P, unsigned).
struct Datapath {
    std::vector<int32_t> mm1W, mm1A, mm1Psum;  // 16 each: int8, int8, int17
    std::vector<int32_t> mm2W, mm2A, mm2Psum;  // 16 each: int8, uint9, int18
};

struct KVBuffers {
    int sel;
    std::vector<int32_t> k;  // 32
    std::vector<int32_t> v;  // 32
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
    TileState getTile() const;
    KVBuffers getKVBuffers() const;
    Datapath getDatapath() const;
    // P tiles written by a NAIVE run (HBM words 768..): n x n probabilities (UQ1.8), row-major. Requires n known.
    std::vector<int32_t> readProbabilities();

private:
    void clockEdge();
    void evalComb();

    VerilatedContext* ctx_;
    Vtiny_attn_top* top_;
    int n_ = 4;
    int qrows_ = 4;
    bool causal_ = false;
    uint32_t cycle_ = 0;
};
