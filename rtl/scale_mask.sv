/* verilator lint_off IMPORTSTAR */  // package import at file scope: Yosys rejects it in the module header
// S (int17) -> Z (Q8.8): Z = sat((S * cfg_scale) >>> (SF - 8)) clipped to +-32767; a masked entry becomes 0x8000.
// Truncating (floor) shift, as in fixed.scale_mask.
import tiny_attn_pkg::*;
module scale_mask
(
    input  logic signed [S_W-1:0]     s,
    input  logic        [SCALE_W-1:0] scale,
    input  logic                      valid,   // 0 = masked (causal)
    output logic signed [Z_W-1:0]     z
);
    localparam int PROD_W = S_W + SCALE_W + 1;

    logic signed [PROD_W-1:0] prod, shifted;

    always_comb begin
        prod    = s * $signed({1'b0, scale});
        shifted = prod >>> (SF - 8);
        if (!valid)                                         z = Z_MASK;
        else if (shifted > PROD_W'(Z_MAX))                  z = Z_MAX;
        else if (shifted < PROD_W'(Z_MIN))                  z = Z_MIN;
        else                                                z = shifted[Z_W-1:0];
    end
endmodule
