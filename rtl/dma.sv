// Burst mover between HBM and a word stream. One burst at a time; the controller waits for `done`.
//   read  (dir = 0): cfg_latency cycles after start, then one word per cycle from rd_addr upward.
//                    The word is delivered on wd_data with its burst index on wd_idx (wd_valid = 1).
//   write (dir = 1): posted, no latency; one word per cycle from src_word (the controller's source is
//                    indexed by wd_idx) written to wr_addr upward.
// Burst timing from the start cycle (cycle 0, start high): read done is visible in cycle L + len + 2;
// write done is visible in cycle len + 1. Pinned in sim/test_dma.py.
module dma #(
    parameter int ADDR_W = 8,
    parameter int LEN_W  = 7
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic [6:0]          cfg_latency,
    input  logic                start,          // one-cycle pulse, ignored while busy
    input  logic                dir,            // 0: HBM -> stream, 1: stream -> HBM
    input  logic [ADDR_W-1:0]   hbm_addr,
    input  logic [LEN_W-1:0]    len,            // words, >= 1

    output logic                done,           // one-cycle pulse, the cycle after the last word

    output logic                rd_en,
    output logic [ADDR_W-1:0]   rd_addr,
    input  logic [31:0]         rd_word,

    output logic                wr_en,
    output logic [ADDR_W-1:0]   wr_addr,
    output logic [31:0]         wr_word,

    output logic                wd_valid,       // read: word available this cycle
    output logic [5:0]          wd_idx,         // word index inside the burst, 0..63 (read) / source index (write)
    output logic [31:0]         wd_data,        // read data
    input  logic [31:0]         src_word        // write data for wd_idx
);
    typedef enum logic [1:0] {D_IDLE, D_WAIT, D_RUN} dstate_t;
    dstate_t st;

    logic                 dir_q;
    logic                 done_q;
    logic [ADDR_W-1:0]    ptr;
    logic [5:0]           idx;          // burst index; wraps only after the last word
    logic [LEN_W-1:0]     cnt;
    logic [6:0]           wcnt;

    logic run;
    assign run = (st == D_RUN);

    assign rd_en    = run && !dir_q;
    assign rd_addr  = ptr;
    assign wr_en    = run && dir_q;
    assign wr_addr  = ptr;
    assign wr_word  = src_word;
    assign wd_valid = rd_en;
    assign wd_idx   = idx;
    assign wd_data  = rd_word;
    assign done     = done_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st     <= D_IDLE;
            dir_q  <= 1'b0;
            done_q <= 1'b0;
            ptr    <= '0;
            idx    <= '0;
            cnt    <= '0;
            wcnt   <= '0;
        end else begin
            done_q <= 1'b0;
            case (st)
                D_IDLE: begin
                    if (start) begin
                        dir_q <= dir;
                        ptr   <= hbm_addr;
                        idx   <= '0;
                        cnt   <= len;
                        wcnt  <= cfg_latency;
                        st    <= dir ? D_RUN : D_WAIT;
                    end
                end
                D_WAIT: begin
                    if (wcnt == 7'd0) st <= D_RUN;
                    else              wcnt <= wcnt - 7'd1;
                end
                D_RUN: begin
                    ptr <= ptr + 1'b1;
                    idx <= idx + 1'b1;
                    cnt <= cnt - 1'b1;
                    if (cnt == LEN_W'(1)) begin
                        st     <= D_IDLE;
                        done_q <= 1'b1;
                    end
                end
                default: st <= D_IDLE;
            endcase
        end
    end
endmodule
