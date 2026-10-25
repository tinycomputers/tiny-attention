// Tile core for one Q block of up to 32 rows (8 row groups of 4), driven by flash_ctrl or naive_ctrl.
//
// Operations (tile_op, see tiny_attn_pkg):
//   OP_FLASH  Per K/V tile: kv_load (4 cycles) puts Kᵀ into MM1 and V into MM2. Then each unmasked row group runs
//             MM1 (A = Q rows, W = Kᵀ) -> scale/mask -> online softmax -> MM2 (A = P, W = V) -> O update.
//             Row state (m, l, alpha, O) is per row, index {group, row-in-group}. After the last tile, blk_norm.
//   OP_NZ     MM1 + scale/mask only: each Z row goes to the output port. No state change.
//   OP_NM     tin (4 Z rows) -> m = max(m, rowmax) per row.
//   OP_NP     tin (4 Z rows) -> P = exp2(Z - m) per row (alpha = 1 by construction), l += sum(P); P rows to the output port.
//   OP_NPV    tin (4 P rows, 9 bit in 16-bit lanes) through MM2 with the loaded V weights; O += P.V with alpha = 1.
//
// Timing of an OP_FLASH group tile (c0 = first cycle after tile_start): Q rows enter MM1 at c0..c0+3, S rows leave
// MM1 at c0+7..c0+10, P rows enter MM2 at c0+8..c0+11, PV rows leave at c0+15..c0+18, tile_done at c0+19.
// tile_start to tile_done is 20 cycles (sim/test_attn_tile.py).
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
    input  logic [2:0]             tile_op,
    input  logic [2:0]             tile_grp,        // row group within the block
    input  logic                   tile_diag,       // this group's diagonal K/V tile (causal mask inside)
    input  logic [4*4*16-1:0]      tin,             // NAIVE input tile: 4 rows x 4 lanes x 16 bit, row-major
    output logic                   tile_busy,
    output logic                   tile_done,

    input  logic                   blk_norm,
    input  logic [5:0]             blk_nrows,       // rows to normalize (4 per group in use)
    output logic                   blk_done,
    output logic                   out_valid,
    output logic [4:0]             out_idx,         // row within the block (or tile row for NZ / NP)
    output logic [4*OUT_W-1:0]     out_row,         // Q8.8 V-LSB values (O), or Z / P lanes

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
    typedef enum logic [1:0] {S_IDLE, S_RUN, S_ROWS, S_NORM} state_t;
    state_t state;

    // ---- per-row state (row = {group, row-in-group})
    logic [4*Q_W-1:0]       q_buf   [32];
    logic signed [Z_W-1:0]  m_q     [32];
    logic [L_W-1:0]         l_q     [32];
    logic [E_W-1:0]         alpha_q [32];
    logic [4*O_W-1:0]       o_q     [32];

    // ---- per-tile values (4 rows of the current group), FLASH only
    logic [4*S_W-1:0]       s_t     [4];
    logic [4*Z_W-1:0]       z_t     [4];
    logic [4*P_W-1:0]       p_t     [4];

    logic [2:0]             cur_grp;
    logic [2:0]             op_q;
    logic                   diag_q, streaming, p_valid;
    logic [1:0]             qcnt, srow, orow;
    logic [4*P_W-1:0]       p_reg;
    logic [5:0]             nrows_q;
    logic [4:0]             nrow;

    logic [4:0] sidx, oidx;
    assign sidx = {cur_grp, srow};
    assign oidx = {cur_grp, orow};

    logic is_flash, is_nz, is_nm, is_np, is_npv, mm1_run;
    assign is_flash = (op_q == OP_FLASH);
    assign is_nz    = (op_q == OP_NZ);
    assign is_nm    = (op_q == OP_NM);
    assign is_np    = (op_q == OP_NP);
    assign is_npv   = (op_q == OP_NPV);
    assign mm1_run  = is_flash || is_nz;

    // ---- tin rows (NAIVE): row i = tin[i*64 +: 64], lane j = bits [j*16 +: 16]
    logic [63:0] tin_row;
    assign tin_row = tin[{srow, 6'b0} +: 64];

    // NPV streams P rows from tin: low 9 bits of each lane
    logic [4*P_W-1:0] npv_row;
    logic [7:0]       npv_base;
    assign npv_base = {qcnt, 6'b0};
    for (genvar j = 0; j < 4; j++) begin : g_npv
        assign npv_row[j*P_W +: P_W] = tin[npv_base + 8'(j*16) +: P_W];
    end

    // ---- MM1: S = Q . Kᵀ
    logic             mm1_c_valid;
    logic [4*S_W-1:0] mm1_c_row;
    mm_engine #(.N(4), .A_W(Q_W), .A_SIGNED(1'b1), .W_W(Q_W), .ACC_W(S_W)) u_mm1 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(kt_row),
        .a_valid(streaming && mm1_run), .a_row(q_buf[{cur_grp, qcnt}]),
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

    // ---- online softmax: FLASH takes Z from MM1; NM and NP take Z from tin
    logic [4*Z_W-1:0]      z_src;
    logic signed [Z_W-1:0] m_new;
    logic [E_W-1:0]        alpha_new;
    logic [4*P_W-1:0]      p_row;
    logic [L_W-1:0]        l_new;
    assign z_src = (is_nm || is_np) ? tin_row : z_row;
    online_softmax u_softmax (
        .z(z_src), .m_old(m_q[sidx]), .l_old(l_q[sidx]),
        .m_new(m_new), .alpha(alpha_new), .p(p_row), .l_new(l_new)
    );

    // P row as 16-bit lanes (NAIVE stores P in HBM as 16-bit lanes; the value is at most 256)
    logic [4*OUT_W-1:0] p_lanes;
    for (genvar j = 0; j < 4; j++) begin : g_plane
        assign p_lanes[j*OUT_W +: OUT_W] = OUT_W'(p_row[j*P_W +: P_W]);
    end

    // ---- MM2: PV = P . V (FLASH: P from the softmax; NPV: P from tin)
    logic                mm2_c_valid;
    logic [4*PV_W-1:0]   mm2_c_row;
    mm_engine #(.N(4), .A_W(P_W), .A_SIGNED(1'b0), .W_W(Q_W), .ACC_W(PV_W)) u_mm2 (
        .clk(clk), .rst_n(rst_n),
        .w_row_valid(kv_load_valid), .w_row_idx(kv_load_idx), .w_row(v_row),
        .a_valid(is_npv ? streaming : p_valid), .a_row(is_npv ? npv_row : p_reg),
        .c_valid(mm2_c_valid), .c_row(mm2_c_row), .busy(),
        .dbg_w(dbg_mm2_w), .dbg_a(dbg_mm2_a), .dbg_psum(dbg_mm2_psum)
    );

    // ---- O update for the row leaving MM2. NPV has alpha = 1 (UQ1.15 1.0 = 32768).
    logic [4*O_W-1:0] o_new;
    logic [E_W-1:0]   alpha_o;
    assign alpha_o = is_npv ? E_W'(32768) : alpha_q[oidx];
    o_accumulator u_oacc (.o_old(o_q[oidx]), .alpha(alpha_o), .pv(mm2_c_row), .o_new(o_new));

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

    assign tile_busy   = (state != S_IDLE);
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
            op_q      <= '0;
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
            p_valid   <= mm1_c_valid && is_flash;

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
                        op_q      <= tile_op;
                        cur_grp   <= tile_grp;
                        diag_q    <= tile_diag;
                        streaming <= (tile_op != OP_NM) && (tile_op != OP_NP);
                        qcnt      <= '0;
                        srow      <= '0;
                        orow      <= '0;
                        state     <= ((tile_op == OP_NM) || (tile_op == OP_NP)) ? S_ROWS : S_RUN;
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

                    if (mm1_c_valid && is_flash) begin
                        s_t[srow]     <= mm1_c_row;
                        z_t[srow]     <= z_row;
                        p_t[srow]     <= p_row;
                        m_q[sidx]     <= m_new;
                        l_q[sidx]     <= l_new;
                        alpha_q[sidx] <= alpha_new;
                        p_reg         <= p_row;
                        srow          <= srow + 2'd1;
                    end

                    if (mm1_c_valid && is_nz) begin
                        out_valid <= 1'b1;
                        out_idx   <= {3'b0, srow};
                        out_row   <= z_row;
                        srow      <= srow + 2'd1;
                        if (srow == 2'd3) begin
                            tile_done <= 1'b1;
                            state     <= S_IDLE;
                        end
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

                S_ROWS: begin
                    // one tile row per cycle: NM updates m, NP updates l and emits P
                    if (is_nm) m_q[sidx] <= m_new;
                    if (is_np) begin
                        l_q[sidx]  <= l_new;
                        out_valid  <= 1'b1;
                        out_idx    <= {3'b0, srow};
                        out_row    <= p_lanes;
                    end
                    srow <= srow + 2'd1;
                    if (srow == 2'd3) begin
                        tile_done <= 1'b1;
                        state     <= S_IDLE;
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
