/* verilator lint_off IMPORTSTAR */  // package import at file scope: Yosys rejects it in the module header
// NAIVE (3-pass safe softmax, FlashAttention paper Algorithm 0) controller. Blocks are 4 rows (one row group).
// Per block qi, the four passes run in order over the block's K/V tiles kj = 0 .. last (last = qi when causal):
//   pass 1  load K tile, weight load, OP_NZ: Z tile -> staging -> DMA write to Z[qi][kj]
//   pass 2  DMA read Z[qi][kj] -> tin, OP_NM: row max
//   pass 3  DMA read Z[qi][kj] -> tin, OP_NP: P tile -> staging -> DMA write to P[qi][kj], l accumulates
//   pass 4  DMA read P[qi][kj] -> tin, load V tile, weight load, OP_NPV: O += P.V
// then normalize the block and store O. Per-row results do not depend on the order of tiles, so this block-by-block
// order produces the same values and the same traffic totals as fixed.naive (which runs pass by pass over all blocks).
//
// No K/V overlap in NAIVE (Phase 5 applies it to FLASH only). Buffer 0 is used for every K/V load.
import tiny_attn_pkg::*;
module naive_ctrl
(
    input  logic                clk,
    input  logic                rst_n,
    input  logic                start,        // one-cycle pulse while idle
    input  logic [3:0]          cfg_nb,       // K/V tiles = sequence length / 4 (1..8)
    input  logic                cfg_causal,

    output logic                busy,
    output logic                done,

    // DMA
    output logic                dma_start,
    output logic                dma_dir,      // 0: read, 1: write
    output logic [HBM_AW-1:0]   dma_addr,
    output logic [6:0]          dma_len,
    output logic [1:0]          dma_dst,      // read destination: 0 = Q, 1 = K, 2 = V, 3 = tin (Z or P tile)
    output logic                dma_buf,
    input  logic                dma_done,

    // tile core
    output logic                blk_clear,
    output logic                kv_load_valid,
    output logic [1:0]          kv_load_idx,
    output logic                kv_sel,
    output logic                tile_start,
    output logic [2:0]          tile_op,
    output logic [2:0]          tile_grp,
    output logic                tile_diag,
    input  logic                tile_done,
    output logic                blk_norm,
    output logic [5:0]          blk_nrows,
    input  logic                blk_done,

    output logic                ev_kv_load,   // K tile loads (pass 1)
    output logic                ev_tile,      // tile pairs (pass 1)

    output logic [4:0]          dbg_state
);
    typedef enum logic [4:0] {
        N_IDLE, N_BLK,
        N_LDQ, N_LDQ_W,
        N_ISSUE, N_WAIT,
        N_WLOAD,
        N_TILE, N_TILE_W,
        N_STORE, N_STORE_W,
        N_NEXT,
        N_NORM, N_NORM_W,
        N_STO, N_STO_W,
        N_DONE
    } nstate_t;
    nstate_t st;

    logic [2:0] qi, kj, p;   // block, K/V tile, pass (1..4)
    logic       step;        // load step inside a pass (pass 4 loads P then V)
    logic [1:0] wcnt;

    logic [3:0] last;
    assign last = cfg_causal ? {1'b0, qi} : (cfg_nb - 4'd1);
    logic last_kj;
    assign last_kj = ({1'b0, kj} == last);

    logic [HBM_AW-1:0] z_addr, p_addr;
    assign z_addr = HBM_AW'(Z_BASE) + HBM_AW'({qi, kj, 3'b000});
    assign p_addr = HBM_AW'(P_BASE) + HBM_AW'({qi, kj, 3'b000});

    assign busy      = (st != N_IDLE);
    assign done      = (st == N_DONE);
    assign dbg_state = st;
    assign dma_buf   = 1'b0;

    // DMA start and descriptor by state
    assign dma_start = (st == N_LDQ) || (st == N_ISSUE) || (st == N_STORE) || (st == N_STO);
    assign dma_dir   = (st == N_STORE) || (st == N_STO);

    always_comb begin
        case (st)
            N_LDQ, N_LDQ_W: begin
                dma_addr = HBM_AW'(Q_BASE) + HBM_AW'({qi, 2'b00});
                dma_len  = 7'd4;
                dma_dst  = 2'd0;
            end
            N_STORE, N_STORE_W: begin
                dma_addr = (p == 3'd1) ? z_addr : p_addr;
                dma_len  = 7'd8;
                dma_dst  = 2'd0;
            end
            N_STO, N_STO_W: begin
                dma_addr = HBM_AW'(O_BASE) + HBM_AW'({qi, 3'b000});
                dma_len  = 7'd8;
                dma_dst  = 2'd0;
            end
            default: begin  // N_ISSUE, N_WAIT: loads of the current pass
                if (p == 3'd1) begin
                    dma_addr = HBM_AW'(K_BASE) + HBM_AW'({kj, 2'b00});
                    dma_len  = 7'd4;
                    dma_dst  = 2'd1;
                end else if (p == 3'd4 && step == 1'b1) begin
                    dma_addr = HBM_AW'(V_BASE) + HBM_AW'({kj, 2'b00});
                    dma_len  = 7'd4;
                    dma_dst  = 2'd2;
                end else if (p == 3'd4) begin
                    dma_addr = p_addr;
                    dma_len  = 7'd8;
                    dma_dst  = 2'd3;
                end else begin  // passes 2 and 3: Z tile
                    dma_addr = z_addr;
                    dma_len  = 7'd8;
                    dma_dst  = 2'd3;
                end
            end
        endcase
    end

    assign blk_clear     = (st == N_BLK);
    assign kv_load_valid = (st == N_WLOAD);
    assign kv_load_idx   = wcnt;
    assign kv_sel        = 1'b0;
    assign tile_start    = (st == N_TILE);
    assign tile_grp      = 3'd0;
    always_comb begin
        case (p)
            3'd1:    tile_op = OP_NZ;
            3'd2:    tile_op = OP_NM;
            3'd3:    tile_op = OP_NP;
            default: tile_op = OP_NPV;
        endcase
    end
    assign tile_diag    = (p == 3'd1) && cfg_causal && (kj == qi);
    assign blk_norm     = (st == N_NORM);
    assign blk_nrows    = 6'd4;
    assign ev_kv_load   = (st == N_ISSUE) && (p == 3'd1);
    assign ev_tile      = (st == N_TILE) && (p == 3'd1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st   <= N_IDLE;
            qi   <= '0;
            kj   <= '0;
            p    <= 3'd1;
            step <= 1'b0;
            wcnt <= '0;
        end else begin
            case (st)
                N_IDLE: if (start) begin
                    qi <= '0;
                    st <= N_BLK;
                end
                N_BLK: begin
                    p    <= 3'd1;
                    kj   <= '0;
                    step <= 1'b0;
                    st   <= N_LDQ;
                end
                N_LDQ:   st <= N_LDQ_W;
                N_LDQ_W: if (dma_done) st <= N_ISSUE;
                N_ISSUE: st <= N_WAIT;
                N_WAIT: begin
                    if (dma_done) begin
                        if (p == 3'd4 && step == 1'b0) begin
                            step <= 1'b1;
                            st   <= N_ISSUE;
                        end else begin
                            step <= 1'b0;
                            wcnt <= '0;
                            st   <= (p == 3'd1 || p == 3'd4) ? N_WLOAD : N_TILE;
                        end
                    end
                end
                N_WLOAD: begin
                    wcnt <= wcnt + 2'd1;
                    if (wcnt == 2'd3) st <= N_TILE;
                end
                N_TILE:   st <= N_TILE_W;
                N_TILE_W: if (tile_done) st <= (p == 3'd1 || p == 3'd3) ? N_STORE : N_NEXT;
                N_STORE:  st <= N_STORE_W;
                N_STORE_W: if (dma_done) st <= N_NEXT;
                N_NEXT: begin
                    if (!last_kj) begin
                        kj <= kj + 3'd1;
                        st <= N_ISSUE;
                    end else if (p == 3'd4) begin
                        st <= N_NORM;
                    end else begin
                        p    <= p + 3'd1;
                        kj   <= '0;
                        step <= 1'b0;
                        st   <= N_ISSUE;
                    end
                end
                N_NORM:   st <= N_NORM_W;
                N_NORM_W: if (blk_done) st <= N_STO;
                N_STO:    st <= N_STO_W;
                N_STO_W: begin
                    if (dma_done) begin
                        if (qi == cfg_nb[2:0] - 3'd1) begin
                            st <= N_DONE;
                        end else begin
                            qi <= qi + 3'd1;
                            st <= N_BLK;
                        end
                    end
                end
                N_DONE: st <= N_IDLE;
                default: st <= N_IDLE;
            endcase
        end
    end
endmodule
