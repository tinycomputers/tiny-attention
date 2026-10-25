// One 4-row Q block against a sequence of 4x4 K/V tiles: MM1 -> scale/mask -> online softmax -> MM2 -> O update,
// then normalization after the last tile. No memory system (Phase 4 adds SRAM/DMA/controller): the testbench
// loads Q, K^T and V rows through the load ports and sequences tiles with tile_start.
//
// Protocol (one tile, no overlap with the next):
//   blk_clear           once per Q block: m = 0x8000, l = 0, O = 0
//   q_load_*            4 rows of the resident Q tile (any time while idle)
//   kv_load_*           4 cycles: row r of K^T (= column r of K) into MM1 and row r of V into MM2
//   tile_start          1 cycle, after the last kv_load; tile_diag = diagonal tile of a causal run, tile_last = final tile
//   ... tile_done       pulse when every row of O has been updated (the dbg_* tile values are valid then)
//   out_valid x4        if tile_last: normalized rows 0..3 (Q8.8 V-LSB), blk_done with row 3
//
// Cycle schedule of one tile (c0 = first cycle after tile_start): Q rows enter MM1 at c0..c0+3, S rows leave MM1 at
// c0+7..c0+10, the softmax stage runs on them, P rows enter MM2 at c0+8..c0+11, PV rows leave MM2 at c0+15..c0+18,
// tile_done at c0+19.
/* verilator lint_off PINCONNECTEMPTY */  // mm_engine.busy is unused: tile_busy comes from the FSM
module attn_tile
    import tiny_attn_pkg::*;
(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [SCALE_W-1:0]     cfg_scale,

    input  logic                   blk_clear,
    input  logic                   q_load_valid,
    input  logic [1:0]             q_load_idx,
    input  logic [4*Q_W-1:0]       q_load_row,

    input  logic                   kv_load_valid,
    input  logic [1:0]             kv_load_idx,
    input  logic [4*Q_W-1:0]       kt_row,
    input  logic [4*Q_W-1:0]       v_row,

    input  logic                   tile_start,
    input  logic                   tile_diag,
    input  logic                   tile_last,
    output logic                   tile_busy,
    output logic                   tile_done,

    output logic                   out_valid,
    output logic [1:0]             out_idx,
    output logic [4*OUT_W-1:0]     out_row,
    output logic                   blk_done,

    // ---- debug: every value below is a real hardware register or wire
    output logic [16*S_W-1:0]      dbg_s_tile,     // S, row-major
    output logic [16*Z_W-1:0]      dbg_z_tile,     // Z
    output logic [16*P_W-1:0]      dbg_p_tile,     // P
    output logic [4*E_W-1:0]       dbg_alpha,      // alpha per row (this tile)
    output logic [4*Z_W-1:0]       dbg_m,          // running max per row
    output logic [4*L_W-1:0]       dbg_l,          // running sum per row
    output logic [16*O_W-1:0]      dbg_o,          // O accumulator
    output logic [1:0]             dbg_srow,       // row the softmax stage handles next
    output logic [16*Q_W-1:0]      dbg_mm1_w,
    output logic [16*Q_W-1:0]      dbg_mm1_a,
    output logic [16*S_W-1:0]      dbg_mm1_psum,
    output logic [16*Q_W-1:0]      dbg_mm2_w,
    output logic [16*P_W-1:0]      dbg_mm2_a,
    output logic [16*PV_W-1:0]     dbg_mm2_psum
);
    typedef enum logic [1:0] {S_IDLE, S_RUN, S_NORM} state_t;
    state_t state;

    // ---- block state
    logic        [4*Q_W-1:0]  q_buf   [4];
    logic signed [Z_W-1:0]    m_q     [4];
    logic        [L_W-1:0]    l_q     [4];
    logic        [E_W-1:0]    alpha_q [4];
    logic        [4*O_W-1:0]  o_q     [4];
    logic        [4*S_W-1:0]  s_q     [4];
    logic        [4*Z_W-1:0]  z_q     [4];
    logic        [4*P_W-1:0]  p_q     [4];

    // ---- tile control
    logic       diag_q, last_q, streaming;
    logic [1:0] qcnt, srow, orow, nrow;

    // ---- MM1: S = Q . K^T   (A = Q rows, weights = K^T)
    logic             mm1_c_valid;
    logic [4*S_W-1:0] mm1_c_row;
    mm_engine #(.N(4), .A_W(Q_W), .A_SIGNED(1'b1), .W_W(Q_W), .ACC_W(S_W)) u_mm1 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(kt_row),
        .a_valid(streaming), .a_row(q_buf[qcnt]),
        .c_valid(mm1_c_valid), .c_row(mm1_c_row), .busy(),
        .dbg_w(dbg_mm1_w), .dbg_a(dbg_mm1_a), .dbg_psum(dbg_mm1_psum)
    );

    // ---- softmax stage on the S row that just left MM1
    logic [4*Z_W-1:0]        z_row;
    logic signed [Z_W-1:0]   m_new;
    logic [E_W-1:0]          alpha_new;
    logic [4*P_W-1:0]        p_row;
    logic [L_W-1:0]          l_new;

    for (genvar j = 0; j < 4; j++) begin : g_scale
        scale_mask u_sm (
            .s     ($signed(mm1_c_row[j*S_W +: S_W])),
            .scale (cfg_scale),
            .valid (!diag_q || (j <= int'(srow))),   // diagonal tile: key j > query row is masked
            .z     (z_row[j*Z_W +: Z_W])
        );
    end

    online_softmax u_softmax (
        .z(z_row), .m_old(m_q[srow]), .l_old(l_q[srow]),
        .m_new(m_new), .alpha(alpha_new), .p(p_row), .l_new(l_new)
    );

    // ---- MM2: PV = P . V   (A = P rows, weights = V)
    logic                p_valid;
    logic [4*P_W-1:0]    p_reg;
    logic                mm2_c_valid;
    logic [4*PV_W-1:0]   mm2_c_row;
    mm_engine #(.N(4), .A_W(P_W), .A_SIGNED(1'b0), .W_W(Q_W), .ACC_W(PV_W)) u_mm2 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(v_row),
        .a_valid(p_valid), .a_row(p_reg),
        .c_valid(mm2_c_valid), .c_row(mm2_c_row), .busy(),
        .dbg_w(dbg_mm2_w), .dbg_a(dbg_mm2_a), .dbg_psum(dbg_mm2_psum)
    );

    // ---- O accumulation for the row leaving MM2
    logic [4*O_W-1:0] o_new;
    o_accumulator u_oacc (.o_old(o_q[orow]), .alpha(alpha_q[orow]), .pv(mm2_c_row), .o_new(o_new));

    // ---- normalization (shared reciprocal, one row per cycle)
    logic [R_W-1:0]  r_n;
    logic [K_W-1:0]  k_n;
    logic [4*OUT_W-1:0] norm_row;
    recip_unit u_recip (.l(l_q[nrow]), .r(r_n), .k(k_n));
    for (genvar c = 0; c < 4; c++) begin : g_norm
        normalize u_n (.o($signed(o_q[nrow][c*O_W +: O_W])), .r(r_n), .k(k_n), .out(norm_row[c*OUT_W +: OUT_W]));
    end

    // ---- sequencing
    assign tile_busy = (state != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            streaming <= 1'b0;
            p_valid   <= 1'b0;
            p_reg     <= '0;
            diag_q    <= 1'b0;
            last_q    <= 1'b0;
            qcnt <= '0; srow <= '0; orow <= '0; nrow <= '0;
            tile_done <= 1'b0;
            out_valid <= 1'b0;
            out_idx   <= '0;
            out_row   <= '0;
            blk_done  <= 1'b0;
            for (int i = 0; i < 4; i++) begin
                q_buf[i] <= '0;
                m_q[i] <= Z_MASK;  l_q[i] <= '0;  alpha_q[i] <= '0;  o_q[i] <= '0;
                s_q[i] <= '0;  z_q[i] <= '0;  p_q[i] <= '0;
            end
        end else begin
            tile_done <= 1'b0;
            out_valid <= 1'b0;
            blk_done  <= 1'b0;
            p_valid   <= mm1_c_valid;

            if (q_load_valid) q_buf[q_load_idx] <= q_load_row;

            if (blk_clear) begin
                for (int i = 0; i < 4; i++) begin
                    m_q[i] <= Z_MASK;  l_q[i] <= '0;  o_q[i] <= '0;
                end
            end

            case (state)
                S_IDLE: begin
                    if (tile_start) begin
                        state     <= S_RUN;
                        diag_q    <= tile_diag;
                        last_q    <= tile_last;
                        streaming <= 1'b1;
                        qcnt <= '0; srow <= '0; orow <= '0;
                    end
                end

                S_RUN: begin
                    if (streaming) begin
                        qcnt <= qcnt + 2'd1;
                        if (qcnt == 2'd3) streaming <= 1'b0;
                    end

                    if (mm1_c_valid) begin
                        s_q[srow]     <= mm1_c_row;
                        z_q[srow]     <= z_row;
                        p_q[srow]     <= p_row;
                        m_q[srow]     <= m_new;
                        l_q[srow]     <= l_new;
                        alpha_q[srow] <= alpha_new;
                        p_reg         <= p_row;
                        srow          <= srow + 2'd1;
                    end

                    if (mm2_c_valid) begin
                        o_q[orow] <= o_new;
                        orow      <= orow + 2'd1;
                        if (orow == 2'd3) begin
                            tile_done <= 1'b1;
                            nrow      <= '0;
                            state     <= last_q ? S_NORM : S_IDLE;
                        end
                    end
                end

                S_NORM: begin
                    out_valid <= 1'b1;
                    out_idx   <= nrow;
                    out_row   <= norm_row;
                    nrow      <= nrow + 2'd1;
                    if (nrow == 2'd3) begin
                        blk_done <= 1'b1;
                        state    <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // ---- debug flattening
    for (genvar i = 0; i < 4; i++) begin : g_dbg
        assign dbg_s_tile[i*4*S_W +: 4*S_W] = s_q[i];
        assign dbg_z_tile[i*4*Z_W +: 4*Z_W] = z_q[i];
        assign dbg_p_tile[i*4*P_W +: 4*P_W] = p_q[i];
        assign dbg_alpha[i*E_W +: E_W]      = alpha_q[i];
        assign dbg_m[i*Z_W +: Z_W]          = m_q[i];
        assign dbg_l[i*L_W +: L_W]          = l_q[i];
        assign dbg_o[i*4*O_W +: 4*O_W]      = o_q[i];
    end
    assign dbg_srow = srow;
endmodule
