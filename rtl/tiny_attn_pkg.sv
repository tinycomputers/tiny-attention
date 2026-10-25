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
    // Tile-core operations (tile_op)
    localparam logic [2:0] OP_FLASH = 3'd0;  // online softmax + O (FlashAttention)
    localparam logic [2:0] OP_NZ    = 3'd1;  // NAIVE pass 1: Z rows out
    localparam logic [2:0] OP_NM    = 3'd2;  // NAIVE pass 2: row max from Z
    localparam logic [2:0] OP_NP    = 3'd3;  // NAIVE pass 3: P rows out, l
    localparam logic [2:0] OP_NPV   = 3'd4;  // NAIVE pass 4: O += P.V

    // HBM layout (word addresses). Q, K, V: one word per row (4 int8). O: two words per row (lanes 0-1, 2-3).
    // Z and P tiles (NAIVE): 8 words each (4 rows x 4 lanes x 16 bit), tile (qi,kj) at base + 64*qi + 8*kj.
    localparam int HBM_AW  = 11;   // 2048 words
    localparam int Q_BASE  = 0;
    localparam int K_BASE  = 32;
    localparam int V_BASE  = 64;
    localparam int O_BASE  = 96;
    localparam int Z_BASE  = 256;
    localparam int P_BASE  = 768;
endpackage
