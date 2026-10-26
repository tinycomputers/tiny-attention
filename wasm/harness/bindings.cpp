// Embind bindings: the JavaScript API of the TinyAttn WebAssembly module.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <vector>

#include "harness.h"

using namespace emscripten;

namespace {
std::vector<int32_t> toI32(const val& arr) { return vecFromJSArray<int32_t>(arr); }

val kvToJs(const KVBuffers& kv) {
    val o = val::object();
    o.set("sel", kv.sel);
    o.set("k", val::array(kv.k.begin(), kv.k.end()));
    o.set("v", val::array(kv.v.begin(), kv.v.end()));
    return o;
}

val tileToJs(const TileState& t) {
    val o = val::object();
    o.set("curGrp", t.curGrp);
    o.set("srow", t.srow);
    o.set("s", val::array(t.s.begin(), t.s.end()));
    o.set("z", val::array(t.z.begin(), t.z.end()));
    o.set("p", val::array(t.p.begin(), t.p.end()));
    o.set("alpha", val::array(t.alpha.begin(), t.alpha.end()));
    o.set("m", val::array(t.m.begin(), t.m.end()));
    o.set("l", val::array(t.l.begin(), t.l.end()));
    o.set("o", val::array(t.o.begin(), t.o.end()));
    return o;
}

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
        .function("isDone", &TinyAttnSim::isDone)
        .function("getKVBuffers", optional_override([](TinyAttnSim& self) { return kvToJs(self.getKVBuffers()); }))
        .function("getTile", optional_override([](TinyAttnSim& self) { return tileToJs(self.getTile()); }))
        .function("readProbabilities", optional_override([](TinyAttnSim& self) {
            auto p = self.readProbabilities();
            return val::array(p.begin(), p.end());
        }));
}
