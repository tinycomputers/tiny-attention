// One online-softmax step for ONE row of a 4-wide S tile (combinational). Matches fixed.online_softmax_step.
//   m_new = max(m_old, rowmax(z));  alpha = exp2(m_old - m_new)
//   P_j   = masked ? 0 : round_to_nearest(exp2(z_j - m_new) / 2^7)   (UQ1.15 -> UQ1.8)
//   l_new = ((l_old * alpha) >> 15) + (sum(P) << 4)
// A masked z (0x8000) never wins the max and contributes P = 0.
module online_softmax
    import tiny_attn_pkg::*;
(
    input  logic        [4*Z_W-1:0]   z,       // 4 logits, element j in bits [j*Z_W +: Z_W]
    input  logic signed [Z_W-1:0]     m_old,
    input  logic        [L_W-1:0]     l_old,
    output logic signed [Z_W-1:0]     m_new,
    output logic        [E_W-1:0]     alpha,
    output logic        [4*P_W-1:0]   p,
    output logic        [L_W-1:0]     l_new
);
    localparam int PS_W = P_W + 2;   // sum of four P values
    localparam int LP_W = L_W + E_W; // l_old * alpha

    logic signed [Z_W-1:0] zmax;
    row_max #(.W(Z_W), .N(4)) u_max (.x(z), .max_o(zmax));

    assign m_new = (zmax > m_old) ? zmax : m_old;

    // alpha = exp2(m_old - m_new): difference is always <= 0
    exp2_unit u_alpha (.x(X_W'(m_old) - X_W'(m_new)), .y(alpha));

    logic [PS_W-1:0] psum;
    logic [LP_W-1:0] lprod;

    always_comb begin
        psum = '0;
        for (int j = 0; j < 4; j++) psum += PS_W'(p[j*P_W +: P_W]);
        lprod = LP_W'(l_old) * LP_W'(alpha);
        l_new = L_W'(lprod >> 15) + L_W'(psum << 4);
    end

    for (genvar j = 0; j < 4; j++) begin : g_lane
        logic signed [Z_W-1:0] zj;
        logic                  valid;
        logic signed [X_W-1:0] x;
        logic        [E_W-1:0] e;
        logic        [E_W:0]   rounded;

        assign zj    = z[j*Z_W +: Z_W];
        assign valid = (zj != Z_MASK);
        assign x     = valid ? (X_W'(zj) - X_W'(m_new)) : '0;

        exp2_unit u_e (.x(x), .y(e));

        assign rounded = (E_W+1)'(e) + (E_W+1)'(64);   // + half an LSB of the >> 7
        assign p[j*P_W +: P_W] = valid ? P_W'(rounded >> 7) : '0;
    end
endmodule
