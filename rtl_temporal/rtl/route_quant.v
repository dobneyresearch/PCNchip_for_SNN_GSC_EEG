`default_nettype none
// =============================================================================
// route_quant — quantise the routed adjoint to the interconnect's message width.
// =============================================================================
// Router-fit refinement (#1): the inter-cell error message must be OUTB=6 bits
// (`delta_bits` in the sim), not the wide within-cell MAC. The sim's exact
// quantiser scales δ by the per-channel max OVER THE WHOLE SEQUENCE (non-causal),
// so streaming hardware uses the causal analog: a running-peak scale (the AGC A
// operator, Part II) feeding a fixed-level quantiser (Q). Message on the wire is
// just the OUTB-bit q; the scale is a slow LOCAL per-channel value (the receiver's
// AGC restores magnitude), never sent per message.
//
//   s     <- max(|value|, s - (s >> LEAK))          (leaky running peak, causal)
//   q     =  round(|value| * L / s) * sign(value),  clamped to +-L,  L = 2^(OUTB-1)-1
//
// (round-half-up via (2*num + s) / (2*s)). Bit-faithful ref: ref/routeq_ref.py.
// =============================================================================

module route_quant #(
    parameter integer VW   = 24,   // input value width (signed, raw routed MAC)
    parameter integer OUTB = 6,    // message width (signed) -> L = 2^(OUTB-1)-1
    parameter integer LEAK = 6     // running-peak decay: s -= s>>LEAK per step
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,
    input  wire                   clr,
    input  wire signed [VW-1:0]   value,   // raw routed error for this channel
    output reg  signed [OUTB-1:0] q,       // OUTB-bit quantised message (the wire value)
    output reg        [VW-1:0]    scale     // local running-peak scale (not sent per msg)
);
    localparam integer L = (1 <<< (OUTB-1)) - 1;   // 31 for OUTB=6

    wire [VW-1:0] av    = value[VW-1] ? (~value + 1'b1) : value;   // |value|
    wire [VW-1:0] s_dec = scale - (scale >> LEAK);                 // leaky decay
    wire [VW-1:0] s_nxt = (av > s_dec) ? av : s_dec;               // running peak (incl. current)
    wire [VW-1:0] s_use = (s_nxt == 0) ? {{(VW-1){1'b0}},1'b1} : s_nxt;

    // q = round_half_up(|value| * L / s_use), clamp to L
    wire [VW+OUTB+1:0] num  = av * L;
    wire [VW+OUTB+1:0] qmag = (2*num + s_use) / (2*s_use);
    wire [OUTB-1:0]    qc   = (qmag > L) ? L[OUTB-1:0] : qmag[OUTB-1:0];
    wire signed [OUTB-1:0] q_s = value[VW-1] ? -$signed({1'b0, qc[OUTB-2:0]})
                                             :  $signed({1'b0, qc[OUTB-2:0]});

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q <= {OUTB{1'b0}}; scale <= {VW{1'b0}};
        end else if (clr) begin
            q <= {OUTB{1'b0}}; scale <= {VW{1'b0}};
        end else if (en) begin
            q <= q_s; scale <= s_nxt;
        end
    end
endmodule
`default_nettype wire
