`default_nettype none
// =============================================================================
// pcn_temporal_tile — the temporal chip at WIDTH: an NO x NI synapse tile.
// =============================================================================
// Replicates the verified single lane (pcn_temporal_top dataflow) across a tile:
//   NO postsyn neurons  -> NO x (lif + adjoint_window)  giving delta_o
//   NI presyn inputs    -> NI x (z-delay line)          giving z_del_i
//   NO*NI synapses      -> E_oi += z_del_i ? delta_o : 0 ; fold -> graded_write -> dW_oi
//
// This proves the ARRAY composition (per-neuron adjoints, shared presyn spike lines,
// per-synapse credit). Resource SHARING (one swept comparator/divider time-multiplexed
// across the tile, as jug_ctrl shares its comparator) is the area optimisation on top —
// here the pieces are replicated for a clean correctness demo. Ports are flattened
// vectors (portable). Bit-faithful ref: ref/tile_ref.py.
// =============================================================================

module pcn_temporal_tile #(
    parameter integer NO   = 2,      // postsyn neurons
    parameter integer NI   = 2,      // presyn inputs
    parameter integer INW  = 12, parameter integer EEW = 12,
    parameter integer MEMW = 20, parameter integer ALPHA = 230, parameter integer ASHIFT = 8,
    parameter integer THRESH = 1024, parameter integer PW = 8, parameter integer SURR = 5,
    parameter integer DW = 16, parameter integer N = 8, parameter integer AWACC = 22,
    parameter integer ECW = 32, parameter integer GEW = 16, parameter integer DWW = 20
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clr,
    input  wire                    en,
    input  wire                    fold,
    input  wire [NO*INW-1:0]       in_cur_flat, // per postsyn neuron
    input  wire [NO*EEW-1:0]       e_err_flat,  // per postsyn neuron
    input  wire [NI-1:0]           z_pre,       // per presyn input
    output wire [NO*NI*ECW-1:0]    E_flat,      // per synapse credit
    output wire [NO*NI*DWW-1:0]    dW_flat      // per synapse graded step
);
    genvar o, i;

    // ── NI presyn z-delay lines (shared across the NO neurons) ──
    wire [NI-1:0] z_del;
    generate for (i = 0; i < NI; i = i + 1) begin : gz
        reg [N-1:0] zsr;
        assign z_del[i] = zsr[N-1];
        always @(posedge clk or negedge rst_n)
            if (!rst_n)      zsr <= {N{1'b0}};
            else if (clr)    zsr <= {N{1'b0}};
            else if (en)     zsr <= {zsr[N-2:0], z_pre[i]};
    end endgenerate

    // ── NO postsyn lanes: lif + gated error + adjoint window ──
    wire signed [AWACC-1:0] delta [0:NO-1];
    generate for (o = 0; o < NO; o = o + 1) begin : go
        wire signed [INW-1:0] in_cur = in_cur_flat[o*INW +: INW];
        wire signed [EEW-1:0] e_err  = e_err_flat[o*EEW +: EEW];
        wire signed [MEMW-1:0] mem; wire s_o; wire [PW:0] psi_o;
        lif_cell #(.MEMW(MEMW), .INW(INW), .ALPHA(ALPHA), .ASHIFT(ASHIFT),
                   .THRESH(THRESH), .PW(PW), .SURR(SURR)) u_lif (
            .clk(clk), .rst_n(rst_n), .en(en), .clr(clr),
            .in_cur(in_cur), .mem(mem), .spike(s_o), .psi(psi_o));
        wire signed [PW+1+EEW:0] pe = $signed({1'b0, psi_o}) * $signed(e_err);
        wire signed [PW+1+EEW:0] pe_shr = pe >>> PW;
        wire signed [DW-1:0] a = pe_shr[DW-1:0];
        adjoint_window #(.DW(DW), .N(N), .ACCW(AWACC)) u_adj (
            .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .a(a), .delta(delta[o]));
    end endgenerate

    // ── NO*NI synapses: credit accumulate + graded write ──
    generate for (o = 0; o < NO; o = o + 1) begin : cro
        for (i = 0; i < NI; i = i + 1) begin : cri
            reg signed [ECW-1:0] E_credit;
            wire signed [ECW-1:0] delta_ext = {{(ECW-AWACC){delta[o][AWACC-1]}}, delta[o]};
            localparam signed [ECW-1:0] SAT_P = (1 <<< (GEW-1)) - 1;
            localparam signed [ECW-1:0] SAT_N = -(1 <<< (GEW-1));
            wire signed [GEW-1:0] e_folded = (E_credit > SAT_P) ? SAT_P[GEW-1:0] :
                                             (E_credit < SAT_N) ? SAT_N[GEW-1:0] : E_credit[GEW-1:0];
            wire signed [DWW-1:0] dW_oi;
            graded_write #(.EW(GEW)) u_gw (
                .clk(clk), .rst_n(rst_n), .en(fold), .clr(clr), .E(e_folded),
                .m(), .v(), .step(dW_oi));
            always @(posedge clk or negedge rst_n)
                if (!rst_n)      E_credit <= {ECW{1'b0}};
                else if (clr)    E_credit <= {ECW{1'b0}};
                else if (en)   begin if (z_del[i]) E_credit <= E_credit + delta_ext; end
                else if (fold)   E_credit <= {ECW{1'b0}};
            assign E_flat [(o*NI+i)*ECW +: ECW] = E_credit;
            assign dW_flat[(o*NI+i)*DWW +: DWW] = dW_oi;
        end
    end endgenerate
endmodule
`default_nettype wire
