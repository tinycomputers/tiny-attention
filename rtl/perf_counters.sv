// Run counters, cleared by `clr` (start of a run). cycles counts busy cycles (state != idle, DONE included).
module perf_counters (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clr,
    input  logic        busy,
    input  logic        ev_tile,
    input  logic        ev_kv_load,
    output logic [31:0] cycles,
    output logic [31:0] tiles,
    output logic [31:0] kv_loads
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycles   <= '0;
            tiles    <= '0;
            kv_loads <= '0;
        end else if (clr) begin
            cycles   <= '0;
            tiles    <= '0;
            kv_loads <= '0;
        end else begin
            if (busy)       cycles   <= cycles + 32'd1;
            if (ev_tile)    tiles    <= tiles + 32'd1;
            if (ev_kv_load) kv_loads <= kv_loads + 32'd1;
        end
    end
endmodule
