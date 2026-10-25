// TinyAttention engine, Phase 4: FLASH-2 schedule over an off-chip (HBM) model.
//
// Host protocol: write Q, K, V rows into HBM with the load port (one word per row: lane c in bits [c*8 +: 8]),
// set the config, pulse start, wait for done, read O back (two words per row, lanes 0-1 then 2-3, int16).
// Load and readback only while idle. Counters are cleared by start and hold their values after done.
module tiny_attn_top
    import tiny_attn_pkg::*;
(
    input  logic                clk,
    input  logic                rst_n,

    // configuration (static for a run)
    input  logic [3:0]          cfg_nb,           // K/V tiles: sequence length / 4 (1..8)
    input  logic [3:0]          cfg_gg,           // row groups per Q block: cfg_qrows / 4 (1, 2, 4, 8, <= cfg_nb)
    input  logic                cfg_causal,
    input  logic [6:0]          cfg_hbm_latency,  // cycles before the first word of a read burst
    input  logic [SCALE_W-1:0]  cfg_scale,        // from the host's quantization

    // run control
    input  logic                start,            // one-cycle pulse while idle
    output logic                busy,
    output logic                done,             // one-cycle pulse

    // host access to HBM (idle only)
    input  logic                ld_en,
    input  logic [HBM_AW-1:0]   ld_addr,
    input  logic [31:0]         ld_word,
    input  logic [HBM_AW-1:0]   o_rd_addr,
    output logic [31:0]         o_rd_word,

    // counters
    output logic [31:0]         hbm_rd_bytes,
    output logic [31:0]         hbm_wr_bytes,
    output logic [31:0]         cycles,
    output logic [31:0]         tiles,
    output logic [31:0]         kv_loads,

    // debug (real registers and wires)
    output logic [4:0]          dbg_ctrl_state,
    output logic                dbg_tile_busy,
    output logic [2:0]          dbg_cur_grp,
    output logic [1:0]          dbg_srow,
    output logic [16*S_W-1:0]   dbg_s_tile,
    output logic [16*Z_W-1:0]   dbg_z_tile,
    output logic [16*P_W-1:0]   dbg_p_tile,
    output logic [4*E_W-1:0]    dbg_alpha,
    output logic [4*Z_W-1:0]    dbg_m,
    output logic [4*L_W-1:0]    dbg_l,
    output logic [16*O_W-1:0]   dbg_o,
    output logic [16*Q_W-1:0]   dbg_mm1_w,
    output logic [16*Q_W-1:0]   dbg_mm1_a,
    output logic [16*S_W-1:0]   dbg_mm1_psum,
    output logic [16*Q_W-1:0]   dbg_mm2_w,
    output logic [16*P_W-1:0]   dbg_mm2_a,
    output logic [16*PV_W-1:0]  dbg_mm2_psum
);
    // ---- controller <-> DMA
    logic                dma_start, dma_dir, dma_done;
    logic [HBM_AW-1:0]   dma_addr;
    logic [6:0]          dma_len;
    logic [1:0]          dma_dst;

    // ---- DMA <-> HBM and stream
    logic                dma_rd_en, dma_wr_en;
    logic [HBM_AW-1:0]   dma_rd_addr, dma_wr_addr;
    logic [31:0]         hbm_rd_word, dma_wr_word;
    logic                wd_valid;
    logic [5:0]          wd_idx;
    logic [31:0]         wd_data, src_word;
    logic [4*OUT_W-1:0]  stg_rdata;

    // ---- controller <-> tile core
    logic                blk_clear, kv_load_valid, tile_start, tile_diag, tile_done, blk_norm, blk_done;
    logic [1:0]          kv_load_idx;
    logic [2:0]          tile_grp;
    logic [5:0]          blk_nrows;
    logic                out_valid;
    logic [4:0]          out_idx;
    logic [4*OUT_W-1:0]  out_row;

    logic                ev_kv_load, ev_tile;

    // ---- DMA read words are routed by destination
    logic                q_load_valid;
    logic [4:0]          q_load_idx;
    logic [4*Q_W-1:0]    q_load_row;
    logic [31:0]         kreg [4];
    logic [31:0]         vreg [4];

    assign q_load_valid = wd_valid && (dma_dst == 2'd0);
    assign q_load_idx   = wd_idx[4:0];
    assign q_load_row   = wd_data;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                kreg[i] <= '0;
                vreg[i] <= '0;
            end
        end else if (wd_valid) begin
            if (dma_dst == 2'd1) kreg[wd_idx[1:0]] <= wd_data;
            if (dma_dst == 2'd2) vreg[wd_idx[1:0]] <= wd_data;
        end
    end

    // Kᵀ row r: element i is K[i][r], i.e. byte r of kreg[i]. V row r is vreg[r].
    logic [4*Q_W-1:0] kt_row, v_row;
    for (genvar i = 0; i < 4; i++) begin : g_kt
        assign kt_row[i*Q_W +: Q_W] = kreg[i][{kv_load_idx, 3'b000} +: Q_W];
    end
    assign v_row = vreg[kv_load_idx];

    // ---- output staging: one 64-bit row per block row; DMA write reads it as two 32-bit words
    sram_bank #(.DEPTH(32), .W(4*OUT_W)) u_stg (
        .clk(clk), .we(out_valid), .waddr(out_idx), .wdata(out_row),
        .raddr(wd_idx[5:1]), .rdata(stg_rdata)
    );
    assign src_word = wd_idx[0] ? stg_rdata[63:32] : stg_rdata[31:0];

    // ---- HBM
    hbm_model #(.ADDR_W(HBM_AW)) u_hbm (
        .clk(clk), .rst_n(rst_n), .clr(start),
        .rd_en(dma_rd_en), .rd_addr(dma_rd_addr), .rd_word(hbm_rd_word),
        .wr_en(dma_wr_en), .wr_addr(dma_wr_addr), .wr_word(dma_wr_word),
        .ld_en(ld_en), .ld_addr(ld_addr), .ld_word(ld_word),
        .host_rd_addr(o_rd_addr), .host_rd_word(o_rd_word),
        .rd_bytes(hbm_rd_bytes), .wr_bytes(hbm_wr_bytes)
    );

    dma #(.ADDR_W(HBM_AW), .LEN_W(7)) u_dma (
        .clk(clk), .rst_n(rst_n), .cfg_latency(cfg_hbm_latency),
        .start(dma_start), .dir(dma_dir), .hbm_addr(dma_addr), .len(dma_len),
        .done(dma_done),
        .rd_en(dma_rd_en), .rd_addr(dma_rd_addr), .rd_word(hbm_rd_word),
        .wr_en(dma_wr_en), .wr_addr(dma_wr_addr), .wr_word(dma_wr_word),
        .wd_valid(wd_valid), .wd_idx(wd_idx), .wd_data(wd_data), .src_word(src_word)
    );

    // ---- control
    flash_ctrl u_ctrl (
        .clk(clk), .rst_n(rst_n), .start(start),
        .cfg_nb(cfg_nb), .cfg_gg(cfg_gg), .cfg_causal(cfg_causal),
        .busy(busy), .done(done),
        .dma_start(dma_start), .dma_dir(dma_dir), .dma_addr(dma_addr), .dma_len(dma_len),
        .dma_dst(dma_dst), .dma_done(dma_done),
        .blk_clear(blk_clear), .kv_load_valid(kv_load_valid), .kv_load_idx(kv_load_idx),
        .tile_start(tile_start), .tile_grp(tile_grp), .tile_diag(tile_diag), .tile_done(tile_done),
        .blk_norm(blk_norm), .blk_nrows(blk_nrows), .blk_done(blk_done),
        .ev_kv_load(ev_kv_load), .ev_tile(ev_tile),
        .dbg_state(dbg_ctrl_state)
    );

    // ---- tile core
    attn_tile u_tile (
        .clk(clk), .rst_n(rst_n), .cfg_scale(cfg_scale),
        .blk_clear(blk_clear),
        .q_load_valid(q_load_valid), .q_load_idx(q_load_idx), .q_load_row(q_load_row),
        .kv_load_valid(kv_load_valid), .kv_load_idx(kv_load_idx), .kt_row(kt_row), .v_row(v_row),
        .tile_start(tile_start), .tile_grp(tile_grp), .tile_diag(tile_diag),
        .tile_busy(dbg_tile_busy), .tile_done(tile_done),
        .blk_norm(blk_norm), .blk_nrows(blk_nrows), .blk_done(blk_done),
        .out_valid(out_valid), .out_idx(out_idx), .out_row(out_row),
        .dbg_cur_grp(dbg_cur_grp), .dbg_srow(dbg_srow),
        .dbg_s_tile(dbg_s_tile), .dbg_z_tile(dbg_z_tile), .dbg_p_tile(dbg_p_tile),
        .dbg_alpha(dbg_alpha), .dbg_m(dbg_m), .dbg_l(dbg_l), .dbg_o(dbg_o),
        .dbg_mm1_w(dbg_mm1_w), .dbg_mm1_a(dbg_mm1_a), .dbg_mm1_psum(dbg_mm1_psum),
        .dbg_mm2_w(dbg_mm2_w), .dbg_mm2_a(dbg_mm2_a), .dbg_mm2_psum(dbg_mm2_psum)
    );

    // ---- counters
    perf_counters u_perf (
        .clk(clk), .rst_n(rst_n), .clr(start), .busy(busy),
        .ev_tile(ev_tile), .ev_kv_load(ev_kv_load),
        .cycles(cycles), .tiles(tiles), .kv_loads(kv_loads)
    );
endmodule
