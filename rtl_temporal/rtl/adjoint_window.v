`default_nettype none
// =============================================================================
// adjoint_window — temporal-credit ADJOINT, the corrected credit core (Paper III).
// =============================================================================
// The validated sim credit (_error_carry, uniform mode) is a windowed sum of the
// readiness-gated error, δ_o[t] = Σ_{k=0}^{N} (ψ_o·e_o)[t+k], which is then paired
// with the CURRENT presyn spike:  ΔW_oi = Σ_t δ_o[t]·z_i[t]  (and δ routes down the
// Wᵀ hop). This module computes the running windowed sum of the gated error a=ψ·e.
//
//   S[t] = Σ_{k=0}^{min(t,N)} a[t-k]            (running sum over an N+1 window)
//
// Implemented as add-new / subtract-old over an (N+1)-deep FIFO (same primitive as
// the jug / eligibility — a leaky-accumulator-of-outer-products, here λ=1 boxcar).
// The TOP aligns time (delay z by N) to realise the forward-window credit; here the
// module is just the verified running sum. NB: this SUPERSEDES elig_buffer.v, which
// gated by the presyn spike inside the window (the paper's simplified eq 3) — the
// wrong place for the gate. Bit-faithful ref: ref/adjoint_ref.py.
// =============================================================================

module adjoint_window #(
    parameter integer DW   = 16,   // gated-error input width a = psi*e (signed)
    parameter integer N    = 8,    // window depth
    parameter integer ACCW = 22    // running-sum width (>= DW + ceil(log2(N+1)))
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,      // advance one timestep
    input  wire                   clr,     // clear window + accumulator
    input  wire signed [DW-1:0]   a,       // psi*e for this neuron this step
    output reg  signed [ACCW-1:0] delta    // windowed adjoint = sum of last N+1 gated errors
);
    reg signed [DW-1:0] fifo [0:N];        // N+1 deep
    integer k;

    wire signed [DW-1:0] a_old = fifo[N];  // term leaving the window

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delta <= {ACCW{1'b0}};
            for (k = 0; k <= N; k = k + 1) fifo[k] <= {DW{1'b0}};
        end else if (clr) begin
            delta <= {ACCW{1'b0}};
            for (k = 0; k <= N; k = k + 1) fifo[k] <= {DW{1'b0}};
        end else if (en) begin
            delta <= delta + {{(ACCW-DW){a[DW-1]}}, a}          // sign-extend both
                           - {{(ACCW-DW){a_old[DW-1]}}, a_old};
            for (k = N; k > 0; k = k - 1) fifo[k] <= fifo[k-1];
            fifo[0] <= a;
        end
    end
endmodule
`default_nettype wire
