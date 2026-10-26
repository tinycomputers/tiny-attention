// FPGA top: the TinyAttention engine behind a UART command interface (8N1, the host sets the baud rate).
//
// Commands (host -> device). Multi-byte fields are little-endian.
//   0x01 CONFIG  nb, gg, flags, latency, scale[3]      (7 bytes)  flags: bit0 causal, bit1 naive, bit2 overlap
//   0x02 LOAD    addr[2], word[4]                      (6 bytes)  writes one HBM word; only while idle
//   0x03 START                                         (0 bytes)  runs the engine; the reply comes when it is done
//   0x04 READ    addr[2]                               (2 bytes)  reads one HBM word; reply is 4 bytes
// Replies (device -> host):
//   0x83 DONE    cycles[4], hbm_rd_bytes[4], hbm_wr_bytes[4]   (13 bytes including the code)
//   0x84 WORD    word[4]                                       (5 bytes including the code)
// Unknown command codes are ignored. Cycles, bytes and tiles follow tiny_attn_top's counters.
/* verilator lint_off IMPORTSTAR */  // package import at file scope: Yosys rejects it in the module header
import tiny_attn_pkg::*;

module fpga_top #(
    parameter int CLKS_PER_BIT = 217    // 25 MHz / 115200 baud
) (
    input  logic clk,
    input  logic rst_n,
    input  logic uart_rx,
    output logic uart_tx
);

    localparam logic [7:0] CMD_CONFIG = 8'h01;
    localparam logic [7:0] CMD_LOAD   = 8'h02;
    localparam logic [7:0] CMD_START  = 8'h03;
    localparam logic [7:0] CMD_READ   = 8'h04;
    localparam logic [7:0] REP_DONE   = 8'h83;
    localparam logic [7:0] REP_WORD   = 8'h84;

    // ---- UART
    logic       rx_valid;
    logic [7:0] rx_byte;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(uart_rx), .valid(rx_valid), .data(rx_byte));

    logic       tx_start, tx_busy;
    logic [7:0] tx_byte;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_byte), .busy(tx_busy), .tx(uart_tx));

    // ---- command decoder
    typedef enum logic [2:0] {S_CMD, S_PAY, S_RUN, S_REPLY} fstate_t;
    fstate_t st;
    logic [7:0] cmd;
    logic [2:0] pay_len;        // payload bytes expected
    logic [2:0] pay_cnt;
    logic [7:0] pbuf [7];
    logic [7:0] rbuf [13];      // reply buffer
    logic [3:0] rlen;           // reply length
    logic [3:0] ridx;           // next reply byte to send

    // ---- engine
    logic [3:0]         cfg_nb, cfg_gg;
    logic               cfg_causal, cfg_naive, cfg_overlap;
    logic [6:0]         cfg_latency;
    logic [SCALE_W-1:0] cfg_scale;
    logic               eng_start, eng_busy, eng_done;
    logic               ld_en;
    logic [HBM_AW-1:0]  ld_addr, o_rd_addr;
    logic [31:0]        ld_word, o_rd_word;
    logic [31:0]        hbm_rd_bytes, hbm_wr_bytes, cycles, tiles, kv_loads;
    logic               settle;             // counters are final one edge after done

    tiny_attn_top u_top (
        .clk(clk), .rst_n(rst_n),
        .cfg_nb(cfg_nb), .cfg_gg(cfg_gg), .cfg_causal(cfg_causal), .cfg_naive(cfg_naive),
        .cfg_overlap(cfg_overlap), .cfg_hbm_latency(cfg_latency), .cfg_scale(cfg_scale),
        .start(eng_start), .busy(eng_busy), .done(eng_done),
        .ld_en(ld_en), .ld_addr(ld_addr), .ld_word(ld_word),
        .o_rd_addr(o_rd_addr), .o_rd_word(o_rd_word),
        .hbm_rd_bytes(hbm_rd_bytes), .hbm_wr_bytes(hbm_wr_bytes),
        .cycles(cycles), .tiles(tiles), .kv_loads(kv_loads),
        .dbg_ctrl_state(), .dbg_tile_busy(), .dbg_cur_grp(), .dbg_srow(),
        .dbg_s_tile(), .dbg_z_tile(), .dbg_p_tile(), .dbg_alpha(), .dbg_m(), .dbg_l(), .dbg_o(),
        .dbg_mm1_w(), .dbg_mm1_a(), .dbg_mm1_psum(), .dbg_mm2_w(), .dbg_mm2_a(), .dbg_mm2_psum(),
        .dbg_kbuf(), .dbg_vbuf(), .dbg_kv_sel(), .dbg_tile_done()
    );

    wire [HBM_AW-1:0] pay_addr = HBM_AW'({pbuf[1], pbuf[0]});
    wire [31:0]       pay_word = {pbuf[5], pbuf[4], pbuf[3], pbuf[2]};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= S_CMD;
            cmd       <= '0;
            pay_len   <= '0;
            pay_cnt   <= '0;
            rlen      <= '0;
            ridx      <= '0;
            cfg_nb    <= 4'd1;
            cfg_gg    <= 4'd1;
            cfg_causal  <= 1'b0;
            cfg_naive   <= 1'b0;
            cfg_overlap <= 1'b0;
            cfg_latency <= '0;
            cfg_scale   <= '0;
            eng_start <= 1'b0;
            ld_en     <= 1'b0;
            ld_addr   <= '0;
            ld_word   <= '0;
            o_rd_addr <= '0;
            settle    <= 1'b0;
            tx_start  <= 1'b0;
            tx_byte   <= '0;
        end else begin
            eng_start <= 1'b0;
            ld_en     <= 1'b0;
            tx_start  <= 1'b0;

            // reply transmitter: send rbuf one byte at a time
            if (ridx != rlen && !tx_busy && !tx_start) begin
                tx_byte  <= rbuf[ridx];
                tx_start <= 1'b1;
                ridx     <= ridx + 4'd1;
            end

            case (st)
                S_CMD: begin
                    if (rx_valid) begin
                        cmd <= rx_byte;
                        case (rx_byte)
                            CMD_CONFIG: begin pay_len <= 3'd7; pay_cnt <= '0; st <= S_PAY; end
                            CMD_LOAD:   begin pay_len <= 3'd6; pay_cnt <= '0; st <= S_PAY; end
                            CMD_READ:   begin pay_len <= 3'd2; pay_cnt <= '0; st <= S_PAY; end
                            CMD_START: begin
                                eng_start <= 1'b1;
                                st        <= S_RUN;
                            end
                            default: st <= S_CMD;
                        endcase
                    end
                end

                S_PAY: begin
                    if (rx_valid) begin
                        pbuf[pay_cnt] <= rx_byte;
                        pay_cnt       <= pay_cnt + 3'd1;
                        if (pay_cnt == pay_len - 3'd1) begin
                            case (cmd)
                                CMD_CONFIG: begin
                                    // pbuf: nb, gg, flags, latency, scale[3] (the last byte is in rx_byte)
                                    cfg_nb      <= pbuf[0][3:0];
                                    cfg_gg      <= pbuf[1][3:0];
                                    cfg_causal  <= pbuf[2][0];
                                    cfg_naive   <= pbuf[2][1];
                                    cfg_overlap <= pbuf[2][2];
                                    cfg_latency <= pbuf[3][6:0];
                                    cfg_scale   <= SCALE_W'({rx_byte, pbuf[5], pbuf[4]});
                                    st          <= S_CMD;
                                end
                                CMD_LOAD: begin
                                    ld_en   <= 1'b1;
                                    ld_addr <= pay_addr;
                                    ld_word <= {rx_byte, pbuf[4], pbuf[3], pbuf[2]};
                                    st      <= S_CMD;
                                end
                                CMD_READ: begin
                                    o_rd_addr <= HBM_AW'({rx_byte, pbuf[0]});
                                    st        <= S_REPLY;
                                    // the READ reply is built one cycle later, once o_rd_word has settled
                                    settle    <= 1'b1;
                                end
                                default: st <= S_CMD;
                            endcase
                        end
                    end
                end

                S_RUN: begin
                    if (eng_done) begin
                        settle <= 1'b1;
                    end
                    if (settle) begin
                        settle <= 1'b0;
                        rbuf[0]  <= REP_DONE;
                        rbuf[1]  <= cycles[7:0];   rbuf[2]  <= cycles[15:8];
                        rbuf[3]  <= cycles[23:16]; rbuf[4]  <= cycles[31:24];
                        rbuf[5]  <= hbm_rd_bytes[7:0];   rbuf[6]  <= hbm_rd_bytes[15:8];
                        rbuf[7]  <= hbm_rd_bytes[23:16]; rbuf[8]  <= hbm_rd_bytes[31:24];
                        rbuf[9]  <= hbm_wr_bytes[7:0];   rbuf[10] <= hbm_wr_bytes[15:8];
                        rbuf[11] <= hbm_wr_bytes[23:16]; rbuf[12] <= hbm_wr_bytes[31:24];
                        rlen     <= 4'd13;
                        ridx     <= '0;
                        st       <= S_REPLY;
                    end
                end

                S_REPLY: begin
                    if (cmd == CMD_READ && settle) begin
                        settle  <= 1'b0;
                        rbuf[0] <= REP_WORD;
                        rbuf[1] <= o_rd_word[7:0];
                        rbuf[2] <= o_rd_word[15:8];
                        rbuf[3] <= o_rd_word[23:16];
                        rbuf[4] <= o_rd_word[31:24];
                        rlen    <= 4'd5;
                        ridx    <= '0;
                    end
                    // back to command mode once the reply has been sent
                    if (ridx == rlen && !tx_busy && !tx_start && !settle) st <= S_CMD;
                end

                default: st <= S_CMD;
            endcase
        end
    end

    // START is a one-cycle pulse; the engine is not busy when the host sends START.
endmodule
