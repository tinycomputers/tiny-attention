// Matmul engine: C = A * W for one stationary N x N weight tile and a stream of A rows.
//
//   load weights:  w_row_valid + w_row_idx + w_row (one weight row per cycle, N cycles, array must be idle)
//   stream A:      a_valid + a_row, one row (N elements) per cycle, any gaps allowed
//   results:       c_valid + c_row, C row i appears LATENCY cycles after A row i was presented
//
// Skew registers delay element r of each A row by r cycles; deskew registers delay output column c by
// N-1-c cycles so a whole C row emerges in one cycle. LATENCY = 2N - 1.
module mm_engine #(
    parameter int N        = 4,
    parameter int A_W      = 8,
    parameter bit A_SIGNED = 1'b1,
    parameter int W_W      = 8,
    parameter int ACC_W    = 17
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  w_row_valid,
    input  logic [$clog2(N)-1:0]  w_row_idx,
    input  logic [N*W_W-1:0]      w_row,

    input  logic                  a_valid,
    input  logic [N*A_W-1:0]      a_row,

    output logic                  c_valid,
    output logic [N*ACC_W-1:0]    c_row,
    output logic                  busy,

    output logic [N*N*W_W-1:0]    dbg_w,
    output logic [N*N*A_W-1:0]    dbg_a,
    output logic [N*N*ACC_W-1:0]  dbg_psum
);
    localparam int LATENCY = 2 * N - 1;

    // ---- weight-row write enables
    logic [N-1:0] w_we_row;
    always_comb begin
        w_we_row = '0;
        if (w_row_valid) w_we_row[w_row_idx] = 1'b1;
    end

    // ---- input skew: element r delayed r cycles; a bubble feeds zeros so psums are unaffected
    logic [N*A_W-1:0] a_gated, a_skewed;
    assign a_gated = a_valid ? a_row : '0;

    for (genvar r = 0; r < N; r++) begin : g_skew
        if (r == 0) begin : g_direct
            assign a_skewed[r*A_W +: A_W] = a_gated[r*A_W +: A_W];
        end else begin : g_delay
            logic [A_W-1:0] dly [r];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < r; i++) dly[i] <= '0;
                end else begin
                    dly[0] <= a_gated[r*A_W +: A_W];
                    for (int i = 1; i < r; i++) dly[i] <= dly[i-1];
                end
            end
            assign a_skewed[r*A_W +: A_W] = dly[r-1];
        end
    end

    // ---- array
    logic [N*ACC_W-1:0] psum_bottom;
    systolic_array #(.N(N), .A_W(A_W), .A_SIGNED(A_SIGNED), .W_W(W_W), .ACC_W(ACC_W)) u_array (
        .clk      (clk),
        .rst_n    (rst_n),
        .w_we_row (w_we_row),
        .w_data   (w_row),
        .a_in     (a_skewed),
        .psum_out (psum_bottom),
        .dbg_w    (dbg_w),
        .dbg_a    (dbg_a),
        .dbg_psum (dbg_psum)
    );

    // ---- output deskew: column c delayed N-1-c cycles
    for (genvar c = 0; c < N; c++) begin : g_deskew
        if (c == N - 1) begin : g_direct
            assign c_row[c*ACC_W +: ACC_W] = psum_bottom[c*ACC_W +: ACC_W];
        end else begin : g_delay
            localparam int D_C = N - 1 - c;
            logic [ACC_W-1:0] dly [D_C];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < D_C; i++) dly[i] <= '0;
                end else begin
                    dly[0] <= psum_bottom[c*ACC_W +: ACC_W];
                    for (int i = 1; i < D_C; i++) dly[i] <= dly[i-1];
                end
            end
            assign c_row[c*ACC_W +: ACC_W] = dly[D_C-1];
        end
    end

    // ---- valid pipeline
    logic [LATENCY-1:0] v_pipe;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_pipe <= '0;
        else        v_pipe <= {v_pipe[LATENCY-2:0], a_valid};
    end
    assign c_valid = v_pipe[LATENCY-1];
    assign busy    = a_valid | (|v_pipe);
endmodule
