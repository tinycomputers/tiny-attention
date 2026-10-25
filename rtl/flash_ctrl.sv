// FLASH-2 controller (no overlap; cfg_overlap_en is Phase 5). Q blocks of G = cfg_qrows/4 row groups, outer loop.
// Per block: clear state, load Q rows, then for each K/V tile kj up to the block's last tile: load K and V (DMA),
// load Kᵀ/V into the engines (4 cycles), run every unmasked row group of the block through the tile core.
// After the last K/V tile: normalize the block's rows, store O (DMA write), next block.
//
// The state sequence is modelled cycle-exactly in sim/golden/schedule.py; tests/test_e2e.py asserts it.
module flash_ctrl
    import tiny_attn_pkg::*;
(
    input  logic                clk,
    input  logic                rst_n,
    input  logic                start,        // one-cycle pulse while idle
    input  logic [3:0]          cfg_nb,       // K/V tiles = sequence length / 4 (1..8)
    input  logic [3:0]          cfg_gg,       // row groups per Q block = cfg_qrows / 4 (1, 2, 4, 8)
    input  logic                cfg_causal,

    output logic                busy,
    output logic                done,         // one-cycle pulse

    // DMA
    output logic                dma_start,
    output logic                dma_dir,      // 0: read, 1: write
    output logic [HBM_AW-1:0]   dma_addr,
    output logic [6:0]          dma_len,
    output logic [1:0]          dma_dst,      // read destination: 0 = Q, 1 = K, 2 = V
    input  logic                dma_done,

    // tile core
    output logic                blk_clear,
    output logic                kv_load_valid,
    output logic [1:0]          kv_load_idx,
    output logic                tile_start,
    output logic [2:0]          tile_grp,
    output logic                tile_diag,
    input  logic                tile_done,
    output logic                blk_norm,
    output logic [5:0]          blk_nrows,
    input  logic                blk_done,

    // performance events
    output logic                ev_kv_load,
    output logic                ev_tile,

    output logic [4:0]          dbg_state
);
    typedef enum logic [4:0] {
        C_IDLE, C_BLK,
        C_LDQ, C_LDQ_W,
        C_LDK, C_LDK_W,
        C_LDV, C_LDV_W,
        C_WLOAD,
        C_TILE, C_TILE_W,
        C_NORM, C_NORM_W,
        C_STO, C_STO_W,
        C_DONE
    } cstate_t;
    cstate_t st;

    logic [3:0] gb;       // first row group of the current block
    logic [3:0] bg;       // row groups in the current block
    logic [3:0] last;     // last K/V tile index of the current block
    logic [3:0] kj;       // current K/V tile
    logic [3:0] gi;       // row group within the block being tiled
    logic [1:0] wcnt;     // weight-load row

    logic [3:0] nb, gg;
    assign nb = cfg_nb;
    assign gg = cfg_gg;

    logic [3:0] rem, bg_c, last_c, g_glob;
    logic       at_end, skip, last_kj;
    always_comb begin
        rem     = nb - gb;
        bg_c    = (rem < gg) ? rem : gg;
        last_c  = cfg_causal ? (gb + bg_c - 4'd1) : (nb - 4'd1);
        g_glob  = gb + gi;
        at_end  = (gi >= bg);
        skip    = cfg_causal && (kj > g_glob);
        last_kj = (kj == last);
    end

    assign busy      = (st != C_IDLE);
    assign done      = (st == C_DONE);
    assign dbg_state = st;

    assign dma_start = (st == C_LDQ) || (st == C_LDK) || (st == C_LDV) || (st == C_STO);
    assign dma_dir   = (st == C_STO);
    assign dma_dst   = ((st == C_LDK) || (st == C_LDK_W)) ? 2'd1 :
                       ((st == C_LDV) || (st == C_LDV_W)) ? 2'd2 : 2'd0;

    always_comb begin
        case (st)
            C_LDK, C_LDK_W: dma_addr = HBM_AW'(K_BASE) + HBM_AW'({kj, 2'b00});
            C_LDV, C_LDV_W: dma_addr = HBM_AW'(V_BASE) + HBM_AW'({kj, 2'b00});
            C_STO, C_STO_W: dma_addr = HBM_AW'(O_BASE) + HBM_AW'({gb, 3'b000});
            default:        dma_addr = HBM_AW'({gb, 2'b00});                  // Q
        endcase
    end

    always_comb begin
        case (st)
            C_LDQ, C_LDQ_W:       dma_len = 7'({bg, 2'b00});                  // 4 words per group
            C_STO, C_STO_W:       dma_len = 7'({bg, 3'b000});                 // 8 words per group
            default:              dma_len = 7'd4;                             // one K or V tile
        endcase
    end

    assign blk_clear     = (st == C_BLK);
    assign kv_load_valid = (st == C_WLOAD);
    assign kv_load_idx   = wcnt;
    assign tile_start    = (st == C_TILE) && !at_end && !skip;
    assign tile_grp      = gi[2:0];
    assign tile_diag     = cfg_causal && (kj == g_glob);
    assign blk_norm      = (st == C_NORM);
    assign blk_nrows     = 6'({bg, 2'b00});
    assign ev_kv_load    = (st == C_LDK);
    assign ev_tile       = tile_start;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st   <= C_IDLE;
            gb   <= '0;
            bg   <= '0;
            last <= '0;
            kj   <= '0;
            gi   <= '0;
            wcnt <= '0;
        end else begin
            case (st)
                C_IDLE: begin
                    if (start) begin
                        gb <= '0;
                        st <= C_BLK;
                    end
                end
                C_BLK: begin
                    bg   <= bg_c;
                    last <= last_c;
                    kj   <= '0;
                    gi   <= '0;
                    st   <= C_LDQ;
                end
                C_LDQ:   st <= C_LDQ_W;
                C_LDQ_W: if (dma_done) st <= C_LDK;
                C_LDK:   st <= C_LDK_W;
                C_LDK_W: if (dma_done) st <= C_LDV;
                C_LDV:   st <= C_LDV_W;
                C_LDV_W: begin
                    if (dma_done) begin
                        wcnt <= '0;
                        st   <= C_WLOAD;
                    end
                end
                C_WLOAD: begin
                    wcnt <= wcnt + 2'd1;
                    if (wcnt == 2'd3) begin
                        gi <= '0;
                        st <= C_TILE;
                    end
                end
                C_TILE: begin
                    if (at_end) begin
                        gi <= '0;
                        if (last_kj) begin
                            st <= C_NORM;
                        end else begin
                            kj <= kj + 4'd1;
                            st <= C_LDK;
                        end
                    end else if (skip) begin
                        gi <= gi + 4'd1;
                    end else begin
                        st <= C_TILE_W;
                    end
                end
                C_TILE_W: begin
                    if (tile_done) begin
                        gi <= gi + 4'd1;
                        st <= C_TILE;
                    end
                end
                C_NORM:   st <= C_NORM_W;
                C_NORM_W: if (blk_done) st <= C_STO;
                C_STO:    st <= C_STO_W;
                C_STO_W: begin
                    if (dma_done) begin
                        if (gb + bg >= nb) begin
                            st <= C_DONE;
                        end else begin
                            gb <= gb + gg;
                            st <= C_BLK;
                        end
                    end
                end
                C_DONE:  st <= C_IDLE;
                default: st <= C_IDLE;
            endcase
        end
    end
endmodule
