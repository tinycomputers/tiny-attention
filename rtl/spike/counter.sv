module counter (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       en,
    output logic [7:0] debug_bus
);

    logic [7:0] count_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count_reg <= 8'd0;
        else if (en)
            count_reg <= count_reg + 8'd1;
    end

    assign debug_bus = count_reg;

endmodule
