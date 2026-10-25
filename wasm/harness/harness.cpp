#include "harness.h"

#include <stdexcept>

namespace {
// HBM layout (rtl/tiny_attn_pkg.sv)
constexpr uint32_t Q_BASE = 0, K_BASE = 32, V_BASE = 64, O_BASE = 96;
}  // namespace

TinyAttnSim::TinyAttnSim() {
    ctx_ = new VerilatedContext;
    top_ = new Vtiny_attn_top(ctx_, "TOP");
    reset();
}

TinyAttnSim::~TinyAttnSim() {
    top_->final();
    delete top_;
    delete ctx_;
}

void TinyAttnSim::evalComb() { top_->eval(); }

void TinyAttnSim::clockEdge() {
    top_->clk = 0;
    evalComb();
    top_->clk = 1;
    evalComb();
    ++cycle_;
}

void TinyAttnSim::reset() {
    top_->start = 0;
    top_->ld_en = 0;
    top_->ld_addr = 0;
    top_->ld_word = 0;
    top_->o_rd_addr = 0;
    top_->cfg_nb = 1;
    top_->cfg_gg = 1;
    top_->cfg_causal = 0;
    top_->cfg_naive = 0;
    top_->cfg_overlap = 0;
    top_->cfg_hbm_latency = 0;
    top_->cfg_scale = 0;
    top_->clk = 0;
    top_->rst_n = 0;
    clockEdge();
    clockEdge();
    top_->rst_n = 1;
    clockEdge();
    cycle_ = 0;
}

void TinyAttnSim::configure(int n, int qrows, bool causal, int mode, bool overlap, int latency, uint32_t scale) {
    if (n < 4 || n > 32 || n % 4 != 0) throw std::invalid_argument("n must be a multiple of 4 in 4..32");
    if (qrows != 4 && qrows != 8 && qrows != 16 && qrows != 32) throw std::invalid_argument("qrows must be 4, 8, 16 or 32");
    if (qrows > n) throw std::invalid_argument("qrows must not exceed n");
    if (latency < 0 || latency > 64) throw std::invalid_argument("latency must be 0..64");
    if (scale >= (1u << 24)) throw std::invalid_argument("scale must fit 24 bits");
    n_ = n;
    qrows_ = qrows;
    top_->cfg_nb = static_cast<uint8_t>(n / 4);
    top_->cfg_gg = static_cast<uint8_t>(qrows / 4);
    top_->cfg_causal = causal ? 1 : 0;
    top_->cfg_naive = mode == 1 ? 1 : 0;
    top_->cfg_overlap = (overlap && mode == 0) ? 1 : 0;
    top_->cfg_hbm_latency = static_cast<uint8_t>(latency);
    top_->cfg_scale = scale;
}

void TinyAttnSim::loadQKV(const std::vector<int32_t>& q, const std::vector<int32_t>& k, const std::vector<int32_t>& v) {
    const size_t want = static_cast<size_t>(n_) * 4;
    if (q.size() != want || k.size() != want || v.size() != want) throw std::invalid_argument("q, k, v must be n*4 values");
    auto pack = [](const std::vector<int32_t>& m, size_t row) {
        uint32_t w = 0;
        for (int c = 0; c < 4; ++c) {
            uint32_t lane = static_cast<uint32_t>(m[row * 4 + c]) & 0xFFu;
            w |= lane << (8 * c);
        }
        return w;
    };
    auto write = [&](uint32_t addr, uint32_t word) {
        top_->ld_en = 1;
        top_->ld_addr = addr;
        top_->ld_word = word;
        clockEdge();
        top_->ld_en = 0;
    };
    for (int i = 0; i < n_; ++i) write(Q_BASE + i, pack(q, i));
    for (int i = 0; i < n_; ++i) write(K_BASE + i, pack(k, i));
    for (int i = 0; i < n_; ++i) write(V_BASE + i, pack(v, i));
}

void TinyAttnSim::start() {
    top_->start = 1;
    clockEdge();
    top_->start = 0;
}

Snapshot TinyAttnSim::step() {
    clockEdge();
    return snapshot();
}

int TinyAttnSim::runToDone(int maxCycles) {
    for (int i = 0; i < maxCycles; ++i) {
        if (top_->done) return i;
        clockEdge();
    }
    return top_->done ? maxCycles : -1;
}

bool TinyAttnSim::isDone() const { return top_->done != 0; }

Snapshot TinyAttnSim::snapshot() const {
    Snapshot s;
    s.cycle = cycle_;
    s.ctrlState = top_->dbg_ctrl_state;
    s.busy = top_->busy != 0;
    s.done = top_->done != 0;
    s.hbmRdBytes = top_->hbm_rd_bytes;
    s.hbmWrBytes = top_->hbm_wr_bytes;
    s.cycles = top_->cycles;
    s.tiles = top_->tiles;
    s.kvLoads = top_->kv_loads;
    return s;
}

std::vector<int32_t> TinyAttnSim::readOutput() {
    std::vector<int32_t> out(static_cast<size_t>(n_) * 4);
    for (int i = 0; i < n_; ++i) {
        for (int half = 0; half < 2; ++half) {
            top_->o_rd_addr = O_BASE + 2 * i + half;
            evalComb();
            uint32_t w = top_->o_rd_word;
            for (int lane = 0; lane < 2; ++lane) {
                int32_t v = static_cast<int32_t>((w >> (16 * lane)) & 0xFFFFu);
                if (v >= 32768) v -= 65536;
                out[static_cast<size_t>(i) * 4 + 2 * half + lane] = v;
            }
        }
    }
    return out;
}

std::vector<uint32_t> TinyAttnSim::readHbm(int addr, int len) {
    std::vector<uint32_t> words;
    words.reserve(len);
    for (int i = 0; i < len; ++i) {
        top_->o_rd_addr = addr + i;
        evalComb();
        words.push_back(top_->o_rd_word);
    }
    return words;
}
