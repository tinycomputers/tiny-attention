// Phase 0 spike: TWO instances of ONE Verilated model in one wasm module, each with its own
// VerilatedContext. L4 (Naive vs Flash) needs this: it runs two TinyAttnSim side by side.
#include "Vcounter.h"
#include "verilated.h"
#include <emscripten/bind.h>

class CounterSim {
    VerilatedContext* ctx;
    Vcounter* top;

public:
    CounterSim() {
        ctx = new VerilatedContext;
        top = new Vcounter(ctx, "TOP");
        top->clk = 0;
        top->rst_n = 1;
        top->en = 0;
        top->eval();
    }
    ~CounterSim() {
        top->final();
        delete top;
        delete ctx;
    }
    void reset() {
        top->rst_n = 0;
        top->clk = 0;
        top->eval();
        top->clk = 1;
        top->eval();
        top->rst_n = 1;
        top->clk = 0;
        top->eval();
    }
    void step() {
        top->en = 1;
        top->clk = 0;
        top->eval();
        top->clk = 1;
        top->eval();
    }
    int getCount() { return static_cast<int>(top->debug_bus); }
};

using namespace emscripten;

EMSCRIPTEN_BINDINGS(tiny_attn_spike) {
    class_<CounterSim>("CounterSim")
        .constructor<>()
        .function("reset", &CounterSim::reset)
        .function("step", &CounterSim::step)
        .function("getCount", &CounterSim::getCount);
}
