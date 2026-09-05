`default_nettype none
// =============================================================================
// readout_grad — cold on-chip read-out gradient (Paper III eq. 6).
// =============================================================================
// The SECOND half of learning, also forwards-only: the classifier weight W_f is
// trained by its OWN gradient rather than an off-chip least-squares fit. Per
// read-out synapse (output i, feature j):
//
//   accumulate (per sample):  E += e_i * f_j        (weight)   |  E += e_i (bias)
//   fold (per batch):         dW  = clamp( (LR_MULT * E) >>> LSH, CLIP )
//                             E  <- 0
//
// e_i = onehot(y)-p_hat is formed in firmware (the soft-max "boss"); this module
// is the outer-product accumulate + the BOOTSTRAPPED bounded write. LR_MULT/2^LSH
// is the read-out learning-rate boost (rr_lr_mult) a COLD head needs to keep up
// with the slowly-consolidating hidden weights. Same accumulate-and-write datapath
// as the hidden layers — the read-out is on the cell array, not in firmware.
//
// Bit-faithful reference: ref/readout_ref.py.  tb_readout_grad.v checks E and dW.
// =============================================================================

module readout_grad #(
    parameter integer EW      = 12,   // output-error width (signed)
    parameter integer FW      = 12,   // feature width (signed)
    parameter integer ACCW    = 32,   // gradient accumulator width (signed)
    parameter integer LR_MULT = 100,  // read-out LR boost numerator
    parameter integer LSH     = 16,   // ... / 2^LSH
    parameter integer CLIP    = 127   // bounded write (8-bit signed code delta)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  acc,       // accumulate one sample this cycle
    input  wire                  fold,      // emit dW and reset E this cycle
    input  wire                  bias_mode, // 1 = bias synapse (accumulate e only)
    input  wire signed [EW-1:0]  e,         // output error for this class
    input  wire signed [FW-1:0]  f,         // leaky feature for this input
    output reg  signed [ACCW-1:0] E,        // running read-out gradient (checked)
    output reg  signed [8:0]     dW         // bounded code step on fold (checked)
);
    // outer-product term, widened before multiply (self-determined-width trap).
    wire signed [EW+FW-1:0] ef   = $signed(e) * $signed(f);
    wire signed [ACCW-1:0]  term = bias_mode ? {{(ACCW-EW){e[EW-1]}}, e}
                                             : {{(ACCW-(EW+FW)){ef[EW+FW-1]}}, ef};

    // bootstrapped bounded write on fold.
    wire signed [ACCW+8:0] prod = $signed(LR_MULT) * $signed(E);
    wire signed [ACCW+8:0] shr  = prod >>> LSH;
    localparam signed [ACCW+8:0] CLIP_P = CLIP;
    localparam signed [ACCW+8:0] CLIP_N = -CLIP;
    wire signed [8:0] step = (shr >  CLIP_P) ?  CLIP[8:0] :
                             (shr <  CLIP_N) ? -CLIP[8:0] : shr[8:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            E <= {ACCW{1'b0}};
            dW <= 9'sd0;
        end else if (acc) begin
            E <= E + term;
        end else if (fold) begin
            dW <= step;
            E  <= {ACCW{1'b0}};
        end
    end
endmodule
`default_nettype wire
