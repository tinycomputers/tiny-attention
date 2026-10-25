// O update for ONE row (combinational), 4 columns. Matches fixed.o_update:
//   o_new = ((o_old * alpha) >>> 15) + (pv <<< 4)       truncating (floor) shift
// pv is one row of MM2 = P.V for the current tile (signed, PV_W bits).
module o_accumulator
    import tiny_attn_pkg::*;
(
    input  logic [4*O_W-1:0]  o_old,
    input  logic [E_W-1:0]    alpha,
    input  logic [4*PV_W-1:0] pv,
    output logic [4*O_W-1:0]  o_new
);
    localparam int PR_W = O_W + E_W + 1;

    for (genvar c = 0; c < 4; c++) begin : g_col
        logic signed [O_W-1:0]  o;
        logic signed [PV_W-1:0] v;
        logic signed [PR_W-1:0] prod;
        logic signed [O_W-1:0]  shifted;   // (o * alpha) >>> 15 never exceeds |o|, so O_W bits suffice

        assign o       = o_old[c*O_W +: O_W];
        assign v       = pv[c*PV_W +: PV_W];
        assign prod    = PR_W'(o) * $signed({1'b0, alpha});
        assign shifted = O_W'(prod >>> 15);
        assign o_new[c*O_W +: O_W] = shifted + (O_W'(v) <<< 4);
    end
endmodule
