// TinyAttention engine: FLASH-2 or NAIVE schedule over an off-chip (HBM) model, optionally with K/V overlap.
//
// Host protocol: write Q, K, V rows into HBM with the load port (one word per row: lane c in bits [c*8 +: 8]).
// Set the config, pulse start, wait for done, read O back (two words per row, lanes 0-1 then 2-3, int16).
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
    input  logic                cfg_naive,        // 0: FLASH, 1: NAIVE
    input  logic                cfg_overlap,      // FLASH only: double-buffered K/V prefetch
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
    output logic [16*PV_W-1:0]  dbg_mm2_psum,
    // K/V buffers: word (buf*4 + row) holds row `row` of buffer `buf` (4 int8 lanes); dbg_kv_sel is the buffer in use
    output logic [2*4*32-1:0]   dbg_kbuf,
    output logic [2*4*32-1:0]   dbg_vbuf,
    output logic                dbg_kv_sel,
    output logic                dbg_tile_done    // one-cycle pulse: a group tile (or NAIVE row group) completed
);
    // ---- controllers (only the one selected by cfg_naive is started)
    logic                f_busy, f_done, n_busy, n_done;
    logic                f_dma_start, f_dma_dir, f_dma_buf, n_dma_start, n_dma_dir, n_dma_buf;
    logic [HBM_AW-1:0]   f_dma_addr, n_dma_addr;
    logic [6:0]          f_dma_len, n_dma_len;
    logic [1:0]          f_dma_dst, n_dma_dst;
    logic                f_blk_clear, n_blk_clear, f_kv_valid, n_kv_valid, f_kv_sel, n_kv_sel;
    logic [1:0]          f_kv_idx, n_kv_idx;
    logic                f_tile_start, n_tile_start, f_tile_diag, n_tile_diag, f_blk_norm, n_blk_norm;
    logic [2:0]          f_tile_op, n_tile_op, f_tile_grp, n_tile_grp;
    logic [5:0]          f_blk_nrows, n_blk_nrows;
    logic                f_ev_kv, n_ev_kv, f_ev_tile, n_ev_tile;
    logic [4:0]          f_dbg, n_dbg;

    // ---- shared DMA and tile core
    logic                dma_start, dma_dir, dma_buf, dma_done;
    logic [HBM_AW-1:0]   dma_addr;
    logic [6:0]          dma_len;
    logic [1:0]          dma_dst;
    logic                blk_clear, kv_load_valid, kv_sel, tile_start, tile_diag, tile_done, blk_norm, blk_done;
    logic [1:0]          kv_load_idx;
    logic [2:0]          tile_op, tile_grp;
    logic [5:0]          blk_nrows;
    logic                ev_kv_load, ev_tile;

    assign dma_start     = cfg_naive ? n_dma_start : f_dma_start;
    assign dma_dir       = cfg_naive ? n_dma_dir   : f_dma_dir;
    assign dma_buf       = cfg_naive ? n_dma_buf   : f_dma_buf;
    assign dma_addr      = cfg_naive ? n_dma_addr  : f_dma_addr;
    assign dma_len       = cfg_naive ? n_dma_len   : f_dma_len;
    assign dma_dst       = cfg_naive ? n_dma_dst   : f_dma_dst;
    assign blk_clear     = cfg_naive ? n_blk_clear : f_blk_clear;
    assign kv_load_valid = cfg_naive ? n_kv_valid  : f_kv_valid;
    assign kv_load_idx   = cfg_naive ? n_kv_idx    : f_kv_idx;
    assign kv_sel        = cfg_naive ? n_kv_sel    : f_kv_sel;
    assign tile_start    = cfg_naive ? n_tile_start : f_tile_start;
    assign tile_op       = cfg_naive ? n_tile_op   : f_tile_op;
    assign tile_grp      = cfg_naive ? n_tile_grp  : f_tile_grp;
    assign tile_diag     = cfg_naive ? n_tile_diag : f_tile_diag;
    assign blk_norm      = cfg_naive ? n_blk_norm  : f_blk_norm;
    assign blk_nrows     = cfg_naive ? n_blk_nrows : f_blk_nrows;
    assign ev_kv_load    = cfg_naive ? n_ev_kv     : f_ev_kv;
    assign ev_tile       = cfg_naive ? n_ev_tile   : f_ev_tile;
    assign busy          = cfg_naive ? n_busy      : f_busy;
    assign done          = cfg_naive ? n_done      : f_done;
    assign dbg_ctrl_state = cfg_naive ? n_dbg      : f_dbg;

    // ---- K/V double buffers and the NAIVE tile input (Z or P tile: 8 words, 4 rows x 4 lanes x 16 bit)
    logic [31:0] kreg [2][4];
    logic [31:0] vreg [2][4];
    for (genvar b = 0; b < 2; b++) begin : g_kvdbg
        for (genvar r = 0; r < 4; r++) begin : g_row
            assign dbg_kbuf[(b*4+r)*32 +: 32] = kreg[b][r];
            assign dbg_vbuf[(b*4+r)*32 +: 32] = vreg[b][r];
        end
    end
    assign dbg_kv_sel = kv_sel;
    assign dbg_tile_done = tile_done;
    logic [31:0] tin_words [8];

    logic                q_load_valid;
    logic [4:0]          q_load_idx;
    logic [4*Q_W-1:0]    q_load_row;
    logic [5:0]          wd_idx;
    logic                wd_valid;
    logic [31:0]         wd_data, src_word;
    logic [4*OUT_W-1:0]  stg_rdata;

    // DMA read words, routed by destination
    assign q_load_valid = wd_valid && (dma_dst == 2'd0);
    assign q_load_idx   = wd_idx[4:0];
    assign q_load_row   = wd_data;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int b = 0; b < 2; b++) begin
                for (int i = 0; i < 4; i++) begin
                    kreg[b][i] <= '0;
                    vreg[b][i] <= '0;
                end
            end
            for (int w = 0; w < 8; w++) tin_words[w] <= '0;
        end else if (wd_valid) begin
            if (dma_dst == 2'd1) kreg[dma_buf][wd_idx[1:0]] <= wd_data;
            if (dma_dst == 2'd2) vreg[dma_buf][wd_idx[1:0]] <= wd_data;
            if (dma_dst == 2'd3) tin_words[wd_idx[2:0]] <= wd_data;
        end
    end

    // Kᵀ row r: element i is K[i][r], i.e. byte r of kreg[buf][i]. V row r is vreg[buf][r].
    logic [4*Q_W-1:0] kt_row, v_row;
    logic [31:0]      kt_src [4];
    for (genvar i = 0; i < 4; i++) begin : g_kt
        assign kt_src[i] = kreg[kv_sel][i];
        assign kt_row[i*Q_W +: Q_W] = kt_src[i][{kv_load_idx, 3'b000} +: Q_W];
    end
    assign v_row = {vreg[kv_sel][kv_load_idx]};

    logic [4*4*16-1:0] tin;
    for (genvar w = 0; w < 8; w++) begin : g_tin
        assign tin[w*32 +: 32] = tin_words[w];
    end

    logic                out_valid;
    logic [4:0]          out_idx;
    logic [4*OUT_W-1:0]  out_row;

    // ---- output staging: one 64-bit row per output row; DMA writes read it as two 32-bit words
    sram_bank #(.DEPTH(32), .W(4*OUT_W)) u_stg (
        .clk(clk), .we(out_valid), .waddr(out_idx), .wdata(out_row),
        .raddr(wd_idx[5:1]), .rdata(stg_rdata)
    );
    assign src_word = wd_idx[0] ? stg_rdata[63:32] : stg_rdata[31:0];

    // ---- HBM
    logic                dma_rd_en, dma_wr_en;
    logic [HBM_AW-1:0]   dma_rd_addr, dma_wr_addr;
    logic [31:0]         hbm_rd_word, dma_wr_word;

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

    // ---- controllers
    flash_ctrl u_fctrl (
        .clk(clk), .rst_n(rst_n), .start(start && !cfg_naive),
        .cfg_nb(cfg_nb), .cfg_gg(cfg_gg), .cfg_causal(cfg_causal), .cfg_overlap(cfg_overlap),
        .busy(f_busy), .done(f_done),
        .dma_start(f_dma_start), .dma_dir(f_dma_dir), .dma_addr(f_dma_addr), .dma_len(f_dma_len),
        .dma_dst(f_dma_dst), .dma_buf(f_dma_buf), .dma_done(dma_done),
        .blk_clear(f_blk_clear), .kv_load_valid(f_kv_valid), .kv_load_idx(f_kv_idx), .kv_sel(f_kv_sel),
        .tile_start(f_tile_start), .tile_op(f_tile_op), .tile_grp(f_tile_grp), .tile_diag(f_tile_diag),
        .tile_done(tile_done), .blk_norm(f_blk_norm), .blk_nrows(f_blk_nrows), .blk_done(blk_done),
        .ev_kv_load(f_ev_kv), .ev_tile(f_ev_tile), .dbg_state(f_dbg)
    );

    naive_ctrl u_nctrl (
        .clk(clk), .rst_n(rst_n), .start(start && cfg_naive),
        .cfg_nb(cfg_nb), .cfg_causal(cfg_causal),
        .busy(n_busy), .done(n_done),
        .dma_start(n_dma_start), .dma_dir(n_dma_dir), .dma_addr(n_dma_addr), .dma_len(n_dma_len),
        .dma_dst(n_dma_dst), .dma_buf(n_dma_buf), .dma_done(dma_done),
        .blk_clear(n_blk_clear), .kv_load_valid(n_kv_valid), .kv_load_idx(n_kv_idx), .kv_sel(n_kv_sel),
        .tile_start(n_tile_start), .tile_op(n_tile_op), .tile_grp(n_tile_grp), .tile_diag(n_tile_diag),
        .tile_done(tile_done), .blk_norm(n_blk_norm), .blk_nrows(n_blk_nrows), .blk_done(blk_done),
        .ev_kv_load(n_ev_kv), .ev_tile(n_ev_tile), .dbg_state(n_dbg)
    );

    // ---- tile core
    attn_tile u_tile (
        .clk(clk), .rst_n(rst_n), .cfg_scale(cfg_scale),
        .blk_clear(blk_clear),
        .q_load_valid(q_load_valid), .q_load_idx(q_load_idx), .q_load_row(q_load_row),
        .kv_load_valid(kv_load_valid), .kv_load_idx(kv_load_idx), .kt_row(kt_row), .v_row(v_row),
        .tile_start(tile_start), .tile_op(tile_op), .tile_grp(tile_grp), .tile_diag(tile_diag), .tin(tin),
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
