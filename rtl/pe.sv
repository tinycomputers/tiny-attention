// Weight-stationary processing element: psum_out <= psum_in + a_in * w, a_out <= a_in (one register each).
// A_SIGNED = 1: int8 activations (MM1, Q). A_SIGNED = 0: unsigned activations (MM2, P is UQ1.8, 9 bits).
module pe #(
    parameter int A_W      = 8,
    parameter bit A_SIGNED = 1'b1,
    parameter int W_W      = 8,
    parameter int ACC_W    = 17
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    w_we,
    input  logic signed [W_W-1:0]   w_in,
    input  logic        [A_W-1:0]   a_in,
    input  logic signed [ACC_W-1:0] psum_in,
    output logic        [A_W-1:0]   a_out,
    output logic signed [ACC_W-1:0] psum_out,
    output logic signed [W_W-1:0]   dbg_w
);
    logic signed [W_W-1:0]   w_reg;
    logic        [A_W-1:0]   a_reg;
    logic signed [ACC_W-1:0] psum_reg;

    logic signed [A_W:0]       a_ext;
    logic signed [A_W+W_W:0]   prod;

    always_comb begin
        a_ext = A_SIGNED ? {a_in[A_W-1], a_in} : {1'b0, a_in};
        prod  = a_ext * w_reg;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_reg    <= '0;
            a_reg    <= '0;
            psum_reg <= '0;
        end else begin
            if (w_we) w_reg <= w_in;
            a_reg    <= a_in;
            psum_reg <= psum_in + ACC_W'(prod);
        end
    end

    assign a_out    = a_reg;
    assign psum_out = psum_reg;
    assign dbg_w    = w_reg;
endmodule
