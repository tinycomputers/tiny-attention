// 8N1 UART. CLKS_PER_BIT = clock / baud (217 for 25 MHz and 115200 baud; small values for simulation).
module uart_rx #(
    parameter int CLKS_PER_BIT = 217
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       rx,
    output logic       valid,      // one-cycle pulse: data is a received byte
    output logic [7:0] data
);
    typedef enum logic [1:0] {R_IDLE, R_START, R_DATA, R_STOP} rstate_t;
    rstate_t st;
    logic [1:0] sync;              // input synchronizer
    logic [15:0] cnt;
    logic [2:0] bit_idx;
    logic [7:0] shreg;

    wire rx_s = sync[1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync  <= 2'b11;
            st    <= R_IDLE;
            cnt   <= '0;
            bit_idx <= '0;
            shreg <= '0;
            valid <= 1'b0;
            data  <= '0;
        end else begin
            sync  <= {sync[0], rx};
            valid <= 1'b0;
            case (st)
                R_IDLE: begin
                    if (!rx_s) begin
                        cnt <= '0;
                        st  <= R_START;
                    end
                end
                R_START: begin                      // sample the middle of the start bit
                    if (cnt == 16'(CLKS_PER_BIT / 2)) begin
                        if (!rx_s) begin
                            cnt     <= '0;
                            bit_idx <= '0;
                            st      <= R_DATA;
                        end else begin
                            st <= R_IDLE;
                        end
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                R_DATA: begin
                    if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                        cnt   <= '0;
                        shreg <= {rx_s, shreg[7:1]};
                        if (bit_idx == 3'd7) st <= R_STOP;
                        bit_idx <= bit_idx + 3'd1;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                R_STOP: begin
                    if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                        valid <= rx_s;
                        data  <= shreg;
                        st    <= R_IDLE;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                default: st <= R_IDLE;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter int CLKS_PER_BIT = 217
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       start,      // one-cycle pulse, only while not busy
    input  logic [7:0] data,
    output logic       busy,
    output logic       tx
);
    typedef enum logic [1:0] {T_IDLE, T_START, T_DATA, T_STOP} tstate_t;
    tstate_t st;
    logic [15:0] cnt;
    logic [2:0] bit_idx;
    logic [7:0] shreg;

    assign busy = (st != T_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st      <= T_IDLE;
            cnt     <= '0;
            bit_idx <= '0;
            shreg   <= '0;
            tx      <= 1'b1;
        end else begin
            case (st)
                T_IDLE: begin
                    tx <= 1'b1;
                    if (start) begin
                        shreg <= data;
                        cnt   <= '0;
                        tx    <= 1'b0;
                        st    <= T_START;
                    end
                end
                T_START: begin
                    if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                        cnt     <= '0;
                        bit_idx <= '0;
                        tx      <= shreg[0];
                        st      <= T_DATA;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                T_DATA: begin
                    if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                        cnt     <= '0;
                        shreg   <= {1'b0, shreg[7:1]};
                        bit_idx <= bit_idx + 3'd1;
                        if (bit_idx == 3'd7) begin
                            tx <= 1'b1;
                            st <= T_STOP;
                        end else begin
                            tx <= shreg[1];
                        end
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                T_STOP: begin
                    if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                        st <= T_IDLE;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end
                default: st <= T_IDLE;
            endcase
        end
    end
endmodule
