/* verilator lint_off IMPORTSTAR */  // package import at file scope: Yosys rejects it in the module header
// 1/l for l >= 1 (unsigned, L_W bits). 1/l ~= r / 2^(16 + k): r is UQ1.16 (0x8000..0x10000), k = floor(log2 l).
// Normalize to a UQ1.15 mantissa, LUT64 seed on its top fraction bits, one Newton-Raphson step.
// Matches fixed.recip (nr_steps = 1). Combinational; the Phase 4 controller registers it as needed.
// l == 0 is outside the contract and returns r = 0, k = 0.
import tiny_attn_pkg::*;
import lut_pkg::*;
module recip_unit
(
    input  logic [L_W-1:0] l,
    output logic [R_W-1:0] r,
    output logic [K_W-1:0] k
);
    logic [15:0]     m16;       // mantissa in [1,2) as UQ1.15
    logic [15:0]     seed;      // UQ0.16
    // a local copy: Yosys does not resolve part-selects of package-scoped parameters
    localparam logic [16*64-1:0] TAB = RECIP_LUT;
    logic [31:0]     mr;        // m16 * seed
    logic [17:0]     t;         // M*r ~ 1.0 as UQ1.16
    logic [17:0]     e;         // 2 - M*r as UQ1.16
    logic [35:0]     er;        // seed * e

    always_comb begin
        k = '0;
        for (int i = 0; i < L_W; i++) begin
            if (l[i]) k = K_W'(i);
        end
        m16  = 16'((l << (L_W - 1 - int'(k))) >> (L_W - 16));  // normalize: MSB to bit L_W-1, keep top 16
        seed = TAB[m16[14:9]*16 +: 16];
        mr   = m16 * seed;
        t    = 18'(mr >> 15);
        e    = 18'h20000 - t;
        er   = seed * e;
        r    = (l == '0) ? '0 : R_W'(er >> 16);
    end
endmodule
