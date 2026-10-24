// Shared parameters and widths. Every width is proven in docs/ARCHITECTURE.md section 2 and mirrored in
// sim/golden/fixed.py; change them there first.
/* verilator lint_off UNUSEDPARAM */  // shared constants; each module uses a subset
package tiny_attn_pkg;
    localparam int D        = 4;   // head dim == array width
    localparam int BR       = 4;   // tile rows
    localparam int BC       = 4;   // tile cols
    localparam int NMAX     = 32;  // max sequence length

    localparam int Q_W      = 8;   // Q, K, V: int8
    localparam int S_W      = 17;  // S = Q.K^T, signed
    localparam int SCALE_W  = 24;  // cfg_scale, UQ4.20
    localparam int SF       = 20;  // fractional bits of cfg_scale
    localparam int Z_W      = 16;  // Z, m: Q8.8 signed
    localparam int X_W      = 17;  // Z - m: Q9.8 signed, always <= 0
    localparam int E_W      = 16;  // exp2 / alpha: UQ1.15
    localparam int P_W      = 9;   // P: UQ1.8, 0..256
    localparam int PV_W     = 18;  // one MM2 tile: P . V, signed
    localparam int L_W      = 18;  // l: unsigned, real x 2^12
    localparam int O_W      = 26;  // O: signed, sum(P.V) x 2^4
    localparam int R_W      = 17;  // reciprocal mantissa: UQ1.16
    localparam int K_W      = 5;   // reciprocal exponent
    localparam int OUT_W    = 16;  // out: Q8.8 signed, V-LSB units

    localparam logic signed [Z_W-1:0] Z_MASK = 16'sh8000;  // masked logit; unmasked Z saturates to +-32767
    localparam logic signed [Z_W-1:0] Z_MAX  = 16'sh7fff;
    localparam logic signed [Z_W-1:0] Z_MIN  = 16'sh8001;
endpackage
