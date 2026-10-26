/* verilator lint_off IMPORTSTAR */  // package import at file scope: Yosys rejects it in the module header
// 2^(x/256) for x <= 0 (Q9.8) -> UQ1.15. LUT256 on the fraction, right shift by the integer part,
// truncating, exactly 0 once the shift reaches 16. Matches fixed.exp2_q (exp_round = False).
// x > 0 is outside the contract; it saturates to 0xFFFF.
import tiny_attn_pkg::*;
import lut_pkg::*;
module exp2_unit
(
    input  logic signed [X_W-1:0] x,
    output logic        [E_W-1:0] y
);
    logic signed [X_W-9:0] n;       // floor(x / 256), -256..0
    logic        [7:0]     f;
    logic signed [X_W-8:0] s;       // -n, 0..256 (one bit wider than n)
    logic        [E_W-1:0] lut;
    // a local copy: Yosys does not resolve part-selects of package-scoped parameters
    localparam logic [16*256-1:0] TAB = EXP2_LUT;

    always_comb begin
        n   = x[X_W-1:8];
        f   = x[7:0];
        s   = -{n[X_W-9], n};            // sign-extend n by one bit, then negate
        lut = TAB[f*16 +: 16];
        if (x > 0)        y = '1;
        else if (s >= 16) y = '0;
        else             y = lut >> s[3:0];
    end
endmodule
