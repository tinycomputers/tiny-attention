// Max of N signed values (combinational). A masked Z (0x8000, the most negative value) never wins
// unless every input is masked.
module row_max #(
    parameter int W = 16,
    parameter int N = 4
) (
    input  logic [N*W-1:0]        x,        // element i in bits [i*W +: W], signed
    output logic signed [W-1:0]   max_o
);
    always_comb begin
        max_o = $signed(x[0 +: W]);
        for (int i = 1; i < N; i++) begin
            if ($signed(x[i*W +: W]) > max_o) max_o = $signed(x[i*W +: W]);
        end
    end
endmodule
