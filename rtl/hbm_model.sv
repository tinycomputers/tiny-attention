// Off-chip memory model: a word array (one 32-bit word per address) with combinational reads, a DMA write
// port, a host load port, and byte counters for the DMA traffic. Latency is modelled by dma.sv, not here.
// The host load port and the DMA write port share the array; the host only writes while the engine is idle.
module hbm_model #(
    parameter int ADDR_W = 8
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clr,            // clear the byte counters (start of a run)

    input  logic                rd_en,          // DMA read: counted
    input  logic [ADDR_W-1:0]   rd_addr,
    output logic [31:0]         rd_word,

    input  logic                wr_en,          // DMA write: counted
    input  logic [ADDR_W-1:0]   wr_addr,
    input  logic [31:0]         wr_word,

    input  logic                ld_en,          // host load: not counted
    input  logic [ADDR_W-1:0]   ld_addr,
    input  logic [31:0]         ld_word,

    input  logic [ADDR_W-1:0]   host_rd_addr,   // host readback: not counted
    output logic [31:0]         host_rd_word,

    output logic [31:0]         rd_bytes,
    output logic [31:0]         wr_bytes
);
    logic [31:0] mem [0:(1 << ADDR_W) - 1];

    assign rd_word      = mem[rd_addr];
    assign host_rd_word = mem[host_rd_addr];

    always_ff @(posedge clk) begin
        if (ld_en)      mem[ld_addr] <= ld_word;
        else if (wr_en) mem[wr_addr] <= wr_word;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_bytes <= '0;
            wr_bytes <= '0;
        end else if (clr) begin
            rd_bytes <= '0;
            wr_bytes <= '0;
        end else begin
            if (rd_en) rd_bytes <= rd_bytes + 32'd4;
            if (wr_en) wr_bytes <= wr_bytes + 32'd4;
        end
    end
endmodule
