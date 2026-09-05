`default_nettype none
// =============================================================================
// pcn_temporal_top — the TEMPORAL CHIP, single synapse lane (Paper III).
// =============================================================================
// Composes the verified temporal modules in the validated forwards-only dataflow
// (DESIGN.md "P5 chip TOP") for one postsyn neuron o and one presyn input i:
//
//   forward:  lif_cell(in_cur) -> spike s_o, readiness psi_o (per t)
//   gated err a_o[t] = (psi_o[t] * e_o[t]) >> PW                     (Q-normalised)
//   adjoint:  adjoint_window(a_o) -> delta_o   (== forward window δ_fwd[t-N])
//   align:    delay presyn spike z_i by N  (z_del) so δ_fwd[t] meets z_i[t]
//   credit:   E_credit += z_del ? delta_o : 0                        (Σ_t δ·z)
//   fold:     graded_write(sat(E_credit)) -> dW ;  E_credit <- 0
//
// A full chip replicates this lane across the 16x16 array (a swept controller like
// jug_ctrl) and adds the analog MAC + Wᵀ transpose route (pcn_transpose.v). This
// lane proves the COMPOSITION/timing. Bit-faithful ref: ref/top_ref.py.
// =============================================================================

module pcn_temporal_top #(
    // lif
    parameter integer MEMW = 20, parameter integer INW = 12,
    parameter integer ALPHA = 230, parameter integer ASHIFT = 8,
    parameter integer THRESH = 1024, parameter integer PW = 8, parameter integer SURR = 5,
    // credit
    parameter integer EEW = 12,     // transported-error width (signed)
    parameter integer DW  = 16,     // gated-error a width (signed)
    parameter integer N   = 8,      // adjoint window depth
    parameter integer AWACC = 22,   // adjoint accumulator width
    parameter integer ECW = 32,     // credit accumulator width (signed)
    // graded write (folded credit saturated to GEW bits)
    parameter integer GEW = 16
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clr,      // start a new sample (clear lif, window, credit)
    input  wire                   en,       // advance one timestep
    input  wire signed [INW-1:0]  in_cur,   // lif input current (W.s)
    input  wire signed [EEW-1:0]  e_err,    // transported error for this neuron this step
    input  wire                   z_pre,    // presynaptic spike this step
    input  wire                   fold,     // fold: push credit through the graded write
    output wire                   spike,    // lif spike (observability)
    output wire       [PW:0]      psi,      // lif readiness
    output reg  signed [ECW-1:0]  E_credit, // accumulated Σ_t δ·z (checked)
    output wire signed [16+3:0]   dW        // graded-write code step on fold (checked)
);
    wire signed [MEMW-1:0] mem;

    // ── forward: LIF (spike + readiness) ──
    lif_cell #(.MEMW(MEMW), .INW(INW), .ALPHA(ALPHA), .ASHIFT(ASHIFT),
               .THRESH(THRESH), .PW(PW), .SURR(SURR)) u_lif (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr),
        .in_cur(in_cur), .mem(mem), .spike(spike), .psi(psi));

    // ── gated error a = (psi * e) >> PW  (psi in Q_PW, unsigned 0..2^PW) ──
    wire signed [PW+1+EEW:0] pe = $signed({1'b0, psi}) * $signed(e_err);
    wire signed [PW+1+EEW:0] pe_shr = pe >>> PW;
    wire signed [DW-1:0]     a = pe_shr[DW-1:0];

    // ── adjoint window: delta = running Σ_{k=0}^{N} a[t-k] (== δ_fwd[t-N]) ──
    wire signed [AWACC-1:0] delta;
    adjoint_window #(.DW(DW), .N(N), .ACCW(AWACC)) u_adj (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .a(a), .delta(delta));

    // ── align: delay z by N so δ_fwd[t] pairs with z[t] ──
    reg [N-1:0] zsr;
    wire z_del = zsr[N-1];

    // ── credit accumulate: E += z_del ? delta : 0 ──
    wire signed [ECW-1:0] delta_ext = {{(ECW-AWACC){delta[AWACC-1]}}, delta};

    // ── fold: saturate credit to GEW and drive the graded write ──
    localparam signed [ECW-1:0] SAT_P = (1 <<< (GEW-1)) - 1;
    localparam signed [ECW-1:0] SAT_N = -(1 <<< (GEW-1));
    wire signed [GEW-1:0] e_folded = (E_credit > SAT_P) ? SAT_P[GEW-1:0] :
                                     (E_credit < SAT_N) ? SAT_N[GEW-1:0] : E_credit[GEW-1:0];
    graded_write #(.EW(GEW)) u_gw (
        .clk(clk), .rst_n(rst_n), .en(fold), .clr(clr), .E(e_folded),
        .m(), .v(), .step(dW));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            E_credit <= {ECW{1'b0}};
            zsr      <= {N{1'b0}};
        end else if (clr) begin
            E_credit <= {ECW{1'b0}};
            zsr      <= {N{1'b0}};
        end else if (en) begin
            zsr <= {zsr[N-2:0], z_pre};
            if (z_del) E_credit <= E_credit + delta_ext;
        end else if (fold) begin
            E_credit <= {ECW{1'b0}};   // reset credit after the fold consumes it
        end
    end
endmodule
`default_nettype wire
