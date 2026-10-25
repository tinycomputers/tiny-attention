// Embind bindings: the JavaScript API of the TinyAttn WebAssembly module.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <vector>

#include "harness.h"

using namespace emscripten;

namespace {
std::vector<int32_t> toI32(const val& arr) { return vecFromJSArray<int32_t>(arr); }

val snapshotToJs(const Snapshot& s) {
    val o = val::object();
    o.set("cycle", s.cycle);
    o.set("ctrlState", s.ctrlState);
    o.set("busy", s.busy);
    o.set("done", s.done);
    o.set("hbmRdBytes", s.hbmRdBytes);
    o.set("hbmWrBytes", s.hbmWrBytes);
    o.set("cycles", s.cycles);
    o.set("tiles", s.tiles);
    o.set("kvLoads", s.kvLoads);
    return o;
}
}  // namespace

EMSCRIPTEN_BINDINGS(tiny_attn) {
    register_vector<int32_t>("VectorInt32");
    register_vector<uint32_t>("VectorUint32");

    class_<TinyAttnSim>("TinyAttnSim")
        .constructor<>()
        .function("reset", &TinyAttnSim::reset)
        .function("configure", &TinyAttnSim::configure)
        .function("loadQKV", optional_override([](TinyAttnSim& self, const val& q, const val& k, const val& v) {
            self.loadQKV(toI32(q), toI32(k), toI32(v));
        }))
        .function("start", &TinyAttnSim::start)
        .function("step", optional_override([](TinyAttnSim& self) { return snapshotToJs(self.step()); }))
        .function("runToDone", &TinyAttnSim::runToDone)
        .function("readOutput", optional_override([](TinyAttnSim& self) {
            auto out = self.readOutput();
            return val::array(out.begin(), out.end());
        }))
        .function("readHbm", optional_override([](TinyAttnSim& self, int addr, int len) {
            auto words = self.readHbm(addr, len);
            return val::array(words.begin(), words.end());
        }))
        .function("snapshot", optional_override([](TinyAttnSim& self) { return snapshotToJs(self.snapshot()); }))
        .function("isDone", &TinyAttnSim::isDone);
}
