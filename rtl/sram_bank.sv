// On-chip buffer: one write port, one read port, read is COMBINATIONAL (distributed-RAM style).
// Phase 4 uses it for the output staging buffer only. Mapping to block RAM needs one prefetch stage on the
// read side; that change belongs to the FPGA task (Phase 9) and is recorded in docs/ARCHITECTURE.md.
module sram_bank #(
    parameter int DEPTH = 32,
    parameter int W     = 64
) (
    input  logic                         clk,
    input  logic                         we,
    input  logic [$clog2(DEPTH)-1:0]     waddr,
    input  logic [W-1:0]                 wdata,
    input  logic [$clog2(DEPTH)-1:0]     raddr,
    output logic [W-1:0]                 rdata
);
    logic [W-1:0] mem [DEPTH];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    assign rdata = mem[raddr];
endmodule
