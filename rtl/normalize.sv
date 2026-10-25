// out = clip_int16( (o * r + 2^(7+k)) >>> (8+k) ): o / l as Q8.8 in V-LSB units, round to nearest, where
// 1/l ~= r / 2^(16+k) comes from recip_unit. One element (combinational). Matches fixed.normalize.
module normalize
    import tiny_attn_pkg::*;
(
    input  logic signed [O_W-1:0]   o,
    input  logic        [R_W-1:0]   r,
    input  logic        [K_W-1:0]   k,
    output logic signed [OUT_W-1:0] out
);
    localparam int W = O_W + R_W + 2;   // product plus rounding constant

    logic signed [W-1:0] prod, sum, shifted;
    logic        [5:0]   sh;

    always_comb begin
        sh      = 6'd8 + 6'(k);
        prod    = W'(o) * $signed({1'b0, r});
        sum     = prod + (W'(1) <<< (sh - 6'd1));
        shifted = sum >>> sh;
        if (shifted > W'(32767))        out = 16'sh7fff;
        else if (shifted < -W'(32768))  out = 16'sh8000;
        else                            out = shifted[OUT_W-1:0];
    end
endmodule
