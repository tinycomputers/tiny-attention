// N x N weight-stationary systolic array. Row r of the array holds weight row W[r][*]; activation element a_r
// enters the left of row r and flows right; partial sums flow down. Column c of the bottom edge produces
// sum_r a_r * W[r][c]. Inputs must already be skewed (a_r delayed r cycles): see mm_engine.sv.
//
// Weight load: pulse w_we_row[r] with the row on w_data (N weights, W_W bits each, column 0 in the low bits).
module systolic_array #(
    parameter int N        = 4,
    parameter int A_W      = 8,
    parameter bit A_SIGNED = 1'b1,
    parameter int W_W      = 8,
    parameter int ACC_W    = 17
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic [N-1:0]               w_we_row,
    input  logic [N*W_W-1:0]           w_data,
    input  logic [N*A_W-1:0]           a_in,       // element r in bits [r*A_W +: A_W]
    output logic [N*ACC_W-1:0]         psum_out,   // column c in bits [c*ACC_W +: ACC_W]
    output logic [N*N*W_W-1:0]         dbg_w,      // PE (r, c) at index r*N + c
    output logic [N*N*A_W-1:0]         dbg_a,
    output logic [N*N*ACC_W-1:0]       dbg_psum
);
    logic        [A_W-1:0]   a_link [N][N+1];
    logic signed [ACC_W-1:0] p_link [N+1][N];

    for (genvar r = 0; r < N; r++) begin : g_row
        assign a_link[r][0] = a_in[r*A_W +: A_W];
    end
    for (genvar c = 0; c < N; c++) begin : g_top
        assign p_link[0][c] = '0;
        assign psum_out[c*ACC_W +: ACC_W] = p_link[N][c];
    end

    for (genvar r = 0; r < N; r++) begin : g_r
        for (genvar c = 0; c < N; c++) begin : g_c
            pe #(.A_W(A_W), .A_SIGNED(A_SIGNED), .W_W(W_W), .ACC_W(ACC_W)) u_pe (
                .clk      (clk),
                .rst_n    (rst_n),
                .w_we     (w_we_row[r]),
                .w_in     (w_data[c*W_W +: W_W]),
                .a_in     (a_link[r][c]),
                .psum_in  (p_link[r][c]),
                .a_out    (a_link[r][c+1]),
                .psum_out (p_link[r+1][c]),
                .dbg_w    (dbg_w[(r*N+c)*W_W +: W_W])
            );
            assign dbg_a[(r*N+c)*A_W +: A_W]       = a_link[r][c+1];
            assign dbg_psum[(r*N+c)*ACC_W +: ACC_W] = p_link[r+1][c];
        end
    end
endmodule
