// Tile core for one Q block of up to 32 rows (8 row groups of 4), driven by flash_ctrl.
//
// Per K/V tile: kv_load (4 cycles) puts Kᵀ into MM1's weights and V into MM2's weights. Then every unmasked
// row group of the block runs once through the pipeline with tile_start(tile_grp):
//   MM1 (A = Q rows, W = Kᵀ) -> scale/mask -> online softmax -> MM2 (A = P rows, W = V) -> O update.
// Row state (m, l, alpha, O) is kept per row in 32-entry arrays; the row index is {group, row-in-group}.
// After the last K/V tile, blk_norm normalizes the first blk_nrows rows and emits them on out_*.
//
// Timing of one group tile (c0 = first cycle after tile_start): Q rows enter MM1 at c0..c0+3, S rows leave MM1
// at c0+7..c0+10, the softmax stage runs on each, P rows enter MM2 at c0+8..c0+11, PV rows leave MM2 at
// c0+15..c0+18, tile_done at c0+19. tile_start to tile_done is 20 cycles (see sim/test_attn_tile.py).
/* verilator lint_off PINCONNECTEMPTY */  // mm_engine.busy is unused: tile_busy comes from the FSM
module attn_tile
    import tiny_attn_pkg::*;
(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic [SCALE_W-1:0]     cfg_scale,

    input  logic                   blk_clear,
    input  logic                   q_load_valid,
    input  logic [4:0]             q_load_idx,      // Q row within the block
    input  logic [4*Q_W-1:0]       q_load_row,

    input  logic                   kv_load_valid,
    input  logic [1:0]             kv_load_idx,     // row r of the K/V tile
    input  logic [4*Q_W-1:0]       kt_row,          // row r of Kᵀ (= column r of K)
    input  logic [4*Q_W-1:0]       v_row,           // row r of V

    input  logic                   tile_start,
    input  logic [2:0]             tile_grp,        // row group within the block
    input  logic                   tile_diag,       // this group's diagonal K/V tile (causal mask inside)
    output logic                   tile_busy,
    output logic                   tile_done,

    input  logic                   blk_norm,
    input  logic [5:0]             blk_nrows,       // rows to normalize (4 per group in use)
    output logic                   blk_done,
    output logic                   out_valid,
    output logic [4:0]             out_idx,         // row within the block
    output logic [4*OUT_W-1:0]     out_row,         // Q8.8 V-LSB values, lane c in bits [c*16 +: 16]

    // ---- debug: real registers and wires. Tile values are for the group of the most recent tile.
    output logic [2:0]             dbg_cur_grp,
    output logic [1:0]             dbg_srow,
    output logic [16*S_W-1:0]      dbg_s_tile,
    output logic [16*Z_W-1:0]      dbg_z_tile,
    output logic [16*P_W-1:0]      dbg_p_tile,
    output logic [4*E_W-1:0]       dbg_alpha,
    output logic [4*Z_W-1:0]       dbg_m,
    output logic [4*L_W-1:0]       dbg_l,
    output logic [16*O_W-1:0]      dbg_o,
    output logic [16*Q_W-1:0]      dbg_mm1_w,
    output logic [16*Q_W-1:0]      dbg_mm1_a,
    output logic [16*S_W-1:0]      dbg_mm1_psum,
    output logic [16*Q_W-1:0]      dbg_mm2_w,
    output logic [16*P_W-1:0]      dbg_mm2_a,
    output logic [16*PV_W-1:0]     dbg_mm2_psum
);
    typedef enum logic [1:0] {S_IDLE, S_RUN, S_NORM} state_t;
    state_t state;

    // ---- per-row state (row = {group, row-in-group})
    logic [4*Q_W-1:0]       q_buf   [32];
    logic signed [Z_W-1:0]  m_q     [32];
    logic [L_W-1:0]         l_q     [32];
    logic [E_W-1:0]         alpha_q [32];
    logic [4*O_W-1:0]       o_q     [32];

    // ---- per-tile values (4 rows of the current group)
    logic [4*S_W-1:0]       s_t     [4];
    logic [4*Z_W-1:0]       z_t     [4];
    logic [4*P_W-1:0]       p_t     [4];

    logic [2:0]             cur_grp;
    logic                   diag_q, streaming, p_valid;
    logic [1:0]             qcnt, srow, orow;
    logic [4*P_W-1:0]       p_reg;
    logic [5:0]             nrows_q;
    logic [4:0]             nrow;

    logic [4:0] sidx, oidx;
    assign sidx = {cur_grp, srow};
    assign oidx = {cur_grp, orow};

    // ---- MM1: S = Q . Kᵀ
    logic             mm1_c_valid;
    logic [4*S_W-1:0] mm1_c_row;
    mm_engine #(.N(4), .A_W(Q_W), .A_SIGNED(1'b1), .W_W(Q_W), .ACC_W(S_W)) u_mm1 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(kt_row),
        .a_valid(streaming), .a_row(q_buf[{cur_grp, qcnt}]),
        .c_valid(mm1_c_valid), .c_row(mm1_c_row), .busy(),
        .dbg_w(dbg_mm1_w), .dbg_a(dbg_mm1_a), .dbg_psum(dbg_mm1_psum)
    );

    // ---- scale and mask, one lane per key column
    logic [4*Z_W-1:0] z_row;
    for (genvar j = 0; j < 4; j++) begin : g_scale
        scale_mask u_sm (
            .s     ($signed(mm1_c_row[j*S_W +: S_W])),
            .scale (cfg_scale),
            .valid (!diag_q || (j <= int'(srow))),  // diagonal tile: key column j > query row is masked
            .z     (z_row[j*Z_W +: Z_W])
        );
    end

    // ---- online softmax for the row leaving MM1
    logic signed [Z_W-1:0] m_new;
    logic [E_W-1:0]        alpha_new;
    logic [4*P_W-1:0]      p_row;
    logic [L_W-1:0]        l_new;
    online_softmax u_softmax (
        .z(z_row), .m_old(m_q[sidx]), .l_old(l_q[sidx]),
        .m_new(m_new), .alpha(alpha_new), .p(p_row), .l_new(l_new)
    );

    // ---- MM2: PV = P . V
    logic                mm2_c_valid;
    logic [4*PV_W-1:0]   mm2_c_row;
    mm_engine #(.N(4), .A_W(P_W), .A_SIGNED(1'b0), .W_W(Q_W), .ACC_W(PV_W)) u_mm2 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(v_row),
        .a_valid(p_valid), .a_row(p_reg),
        .c_valid(mm2_c_valid), .c_row(mm2_c_row), .busy(),
        .dbg_w(dbg_mm2_w), .dbg_a(dbg_mm2_a), .dbg_psum(dbg_mm2_psum)
    );

    // ---- O update for the row leaving MM2 (alpha of that row was stored by the softmax stage)
    logic [4*O_W-1:0] o_new;
    o_accumulator u_oacc (.o_old(o_q[oidx]), .alpha(alpha_q[oidx]), .pv(mm2_c_row), .o_new(o_new));

    // ---- normalization: one row per cycle, shared reciprocal
    logic [R_W-1:0]     r_n;
    logic [K_W-1:0]     k_n;
    logic [4*OUT_W-1:0] norm_row;
    recip_unit u_recip (.l(l_q[nrow]), .r(r_n), .k(k_n));
    for (genvar c = 0; c < 4; c++) begin : g_norm
        normalize u_n (
            .o($signed(o_q[nrow][c*O_W +: O_W])), .r(r_n), .k(k_n),
            .out(norm_row[c*OUT_W +: OUT_W])
        );
    end

    assign tile_busy = (state != S_IDLE);
    assign dbg_cur_grp = cur_grp;
    assign dbg_srow    = srow;

    for (genvar i = 0; i < 4; i++) begin : g_dbg
        assign dbg_s_tile[i*4*S_W +: 4*S_W] = s_t[i];
        assign dbg_z_tile[i*4*Z_W +: 4*Z_W] = z_t[i];
        assign dbg_p_tile[i*4*P_W +: 4*P_W] = p_t[i];
        assign dbg_alpha[i*E_W +: E_W]      = alpha_q[{cur_grp, 2'(i)}];
        assign dbg_m[i*Z_W +: Z_W]          = m_q[{cur_grp, 2'(i)}];
        assign dbg_l[i*L_W +: L_W]          = l_q[{cur_grp, 2'(i)}];
        assign dbg_o[i*4*O_W +: 4*O_W]      = o_q[{cur_grp, 2'(i)}];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            streaming <= 1'b0;
            p_valid   <= 1'b0;
            p_reg     <= '0;
            cur_grp   <= '0;
            diag_q    <= 1'b0;
            qcnt      <= '0;
            srow      <= '0;
            orow      <= '0;
            nrows_q   <= '0;
            nrow      <= '0;
            tile_done <= 1'b0;
            blk_done  <= 1'b0;
            out_valid <= 1'b0;
            out_idx   <= '0;
            out_row   <= '0;
            for (int i = 0; i < 32; i++) begin
                q_buf[i]   <= '0;
                m_q[i]     <= Z_MASK;
                l_q[i]     <= '0;
                alpha_q[i] <= '0;
                o_q[i]     <= '0;
            end
            for (int i = 0; i < 4; i++) begin
                s_t[i] <= '0;
                z_t[i] <= '0;
                p_t[i] <= '0;
            end
        end else begin
            tile_done <= 1'b0;
            blk_done  <= 1'b0;
            out_valid <= 1'b0;
            p_valid   <= mm1_c_valid;

            if (q_load_valid) q_buf[q_load_idx] <= q_load_row;

            if (blk_clear) begin
                for (int i = 0; i < 32; i++) begin
                    m_q[i] <= Z_MASK;
                    l_q[i] <= '0;
                    o_q[i] <= '0;
                end
            end

            case (state)
                S_IDLE: begin
                    if (tile_start) begin
                        state     <= S_RUN;
                        cur_grp   <= tile_grp;
                        diag_q    <= tile_diag;
                        streaming <= 1'b1;
                        qcnt      <= '0;
                        srow      <= '0;
                        orow      <= '0;
                    end else if (blk_norm) begin
                        state   <= S_NORM;
                        nrows_q <= blk_nrows;
                        nrow    <= '0;
                    end
                end

                S_RUN: begin
                    if (streaming) begin
                        qcnt <= qcnt + 2'd1;
                        if (qcnt == 2'd3) streaming <= 1'b0;
                    end

                    if (mm1_c_valid) begin
                        s_t[srow]     <= mm1_c_row;
                        z_t[srow]     <= z_row;
                        p_t[srow]     <= p_row;
                        m_q[sidx]     <= m_new;
                        l_q[sidx]     <= l_new;
                        alpha_q[sidx] <= alpha_new;
                        p_reg         <= p_row;
                        srow          <= srow + 2'd1;
                    end

                    if (mm2_c_valid) begin
                        o_q[oidx] <= o_new;
                        orow      <= orow + 2'd1;
                        if (orow == 2'd3) begin
                            tile_done <= 1'b1;
                            state     <= S_IDLE;
                        end
                    end
                end

                S_NORM: begin
                    out_valid <= 1'b1;
                    out_idx   <= nrow;
                    out_row   <= norm_row;
                    nrow      <= nrow + 5'd1;
                    if (6'(nrow) + 6'd1 == nrows_q) begin
                        blk_done <= 1'b1;
                        state    <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
