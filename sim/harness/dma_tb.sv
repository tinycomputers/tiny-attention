// Test harness for dma.sv: one HBM model, one DMA, one staging buffer (write source). Nothing else.
module dma_tb (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [6:0]  cfg_latency,
    input  logic        start,
    input  logic        dir,
    input  logic [7:0]  hbm_addr,
    input  logic [6:0]  len,
    output logic        done,
    output logic        wd_valid,
    output logic [5:0]  wd_idx,
    output logic [31:0] wd_data,
    input  logic        ld_en,
    input  logic [7:0]  ld_addr,
    input  logic [31:0] ld_word,
    input  logic [7:0]  host_rd_addr,
    output logic [31:0] host_rd_word,
    input  logic        stg_we,
    input  logic [5:0]  stg_waddr,
    input  logic [31:0] stg_wdata,
    output logic [31:0] rd_bytes,
    output logic [31:0] wr_bytes
);
    logic        rd_en, wr_en;
    logic [7:0]  rd_addr, wr_addr;
    logic [31:0] rd_word, wr_word, src_word;

    hbm_model #(.ADDR_W(8)) u_hbm (
        .clk(clk), .rst_n(rst_n), .clr(start),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_word(rd_word),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_word(wr_word),
        .ld_en(ld_en), .ld_addr(ld_addr), .ld_word(ld_word),
        .host_rd_addr(host_rd_addr), .host_rd_word(host_rd_word),
        .rd_bytes(rd_bytes), .wr_bytes(wr_bytes)
    );

    dma #(.ADDR_W(8), .LEN_W(7)) u_dma (
        .clk(clk), .rst_n(rst_n), .cfg_latency(cfg_latency),
        .start(start), .dir(dir), .hbm_addr(hbm_addr), .len(len),
        .done(done),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_word(rd_word),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_word(wr_word),
        .wd_valid(wd_valid), .wd_idx(wd_idx), .wd_data(wd_data), .src_word(src_word)
    );

    sram_bank #(.DEPTH(64), .W(32)) u_stg (
        .clk(clk), .we(stg_we), .waddr(stg_waddr), .wdata(stg_wdata),
        .raddr(wd_idx), .rdata(src_word)
    );
endmodule
