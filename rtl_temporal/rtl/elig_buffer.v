`default_nettype none
// =============================================================================
// elig_buffer — per-synapse temporal-credit eligibility (Paper III eq. 3).
// =============================================================================
// The temporal-CREDIT role of the leaky-accumulator primitive: a shallow
// n-deep local buffer that forms this synapse's own temporal adjoint, forward
// in time — the "8-step buffer that stands in for a 200-step BPTT unroll".
//
//   g_t = sum_{k=0}^{N-1} delta_{t-k} * s_{t-k}          (windowed eligibility)
//
// The presynaptic term s is a SPIKE (1 bit), so delta*s is delta when the
// presynapse fired this step, else 0. The window is a HARD n-step box (eq. 3,
// uniform weights, cutoff at N) — realised as a running sum with an
// add-new / subtract-old update over an N-deep FIFO of the gated errors:
//
//   p_t   = s_t ? delta_t : 0
//   g_t   = g_{t-1} + p_t - p_{t-N}                       (slide the window)
//
// This is FORWARD-MODE (RTRL-family / SnAp-n): no error ever flows backward,
// no unrolled history is held — only the last N gated errors. Per-layer buffers
// COMPOSE down the stack, so N~8 suffices (Paper III §3.3; depth-flat 6..12).
//
// Bit-faithful reference: ref/elig_ref.py.  tb_elig_buffer.v checks g bit-exact.
// =============================================================================

module elig_buffer #(
    parameter integer DW   = 12,   // transported-error width (signed)
    parameter integer N    = 8,    // window depth (the "8-step buffer")
    parameter integer ACCW = 18    // running-sum width (>= DW + ceil(log2 N))
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,      // advance one timestep
    input  wire                   clr,     // clear buffer + accumulator (start of sequence)
    input  wire signed [DW-1:0]   delta,   // transported error this step (postsyn)
    input  wire                   spk,     // presynaptic spike this step (1 bit)
    output reg  signed [ACCW-1:0] g        // windowed eligibility = sum of last N gated errors
);
    // N-deep FIFO of the gated errors p = spk ? delta : 0.
    reg signed [DW-1:0] fifo [0:N-1];
    integer k;

    wire signed [DW-1:0] p_new = spk ? delta : {DW{1'b0}};
    wire signed [DW-1:0] p_old = fifo[N-1];              // the term leaving the window (p_{t-N})

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g <= {ACCW{1'b0}};
            for (k = 0; k < N; k = k + 1) fifo[k] <= {DW{1'b0}};
        end else if (clr) begin
            g <= {ACCW{1'b0}};
            for (k = 0; k < N; k = k + 1) fifo[k] <= {DW{1'b0}};
        end else if (en) begin
            g <= g + {{(ACCW-DW){p_new[DW-1]}}, p_new}    // sign-extend both
                   - {{(ACCW-DW){p_old[DW-1]}}, p_old};
            for (k = N-1; k > 0; k = k - 1) fifo[k] <= fifo[k-1];
            fifo[0] <= p_new;
        end
    end
endmodule
`default_nettype wire
