`default_nettype none
// =============================================================================
// transpose_route — the Wᵀ hop: route the adjoint DOWN a layer (Paper III).
// =============================================================================
// Deep forwards-only credit: the layer-l adjoint δ_o[t] becomes the error of
// layer l-1 by the transpose of l's own weights (a forward-directed message, not
// a reverse analog channel):
//
//   e_{l-1,i}[t] = sat( (Σ_o δ_o[t] · W_oi) >>> RSH ,  EEW )
//
// A per-presyn MAC over the postsyn adjoints. Combinational (would be the existing
// swept transpose engine `pcn_transpose.v` in the full chip; here a small tile MAC
// for the credit datapath). Scaling RSH+saturate stands in for the sim's
// rms-preserving normalise (exact match = P6-final). Bit-faithful ref: ref/transpose_ref.py.
// =============================================================================

module transpose_route #(
    parameter integer NO    = 2,     // postsyn (adjoint) count
    parameter integer NI    = 2,     // presyn (error out) count
    parameter integer AWACC = 22,    // adjoint width (signed)
    parameter integer WW    = 8,     // weight code width (signed)
    parameter integer EEW   = 12,    // routed-error width (signed)
    parameter integer RSH   = 16     // post-MAC right shift (scale)
) (
    input  wire [NO*AWACC-1:0]  delta_flat,  // δ_o
    input  wire [NO*NI*WW-1:0]  W_flat,      // W_oi (row-major: o*NI+i)
    output wire [NI*EEW-1:0]    e_out_flat   // e_{l-1,i}
);
    localparam integer ACW = AWACC + WW + 8;               // MAC accumulator width
    localparam signed [EEW-1:0] EMAX = (1 <<< (EEW-1)) - 1;
    localparam signed [EEW-1:0] EMIN = -(1 <<< (EEW-1));

    genvar i;
    generate for (i = 0; i < NI; i = i + 1) begin : gi
        reg signed [ACW-1:0] acc;
        integer o;
        always @* begin
            acc = {ACW{1'b0}};
            for (o = 0; o < NO; o = o + 1)
                acc = acc + $signed(delta_flat[o*AWACC +: AWACC])
                          * $signed(W_flat[(o*NI + i)*WW +: WW]);
        end
        wire signed [ACW-1:0] sc = acc >>> RSH;
        wire signed [EEW-1:0] sat = (sc > EMAX) ? EMAX :
                                    (sc < EMIN) ? EMIN : sc[EEW-1:0];
        assign e_out_flat[i*EEW +: EEW] = sat;
    end endgenerate
endmodule
`default_nettype wire
