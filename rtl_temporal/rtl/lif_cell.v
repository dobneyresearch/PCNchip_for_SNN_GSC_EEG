`default_nettype none
// =============================================================================
// lif_cell — one leaky integrate-and-fire neuron (DIGITAL membrane).
// =============================================================================
// The temporal-coding role of the ONE leaky-accumulator primitive (Paper III).
// Structurally identical to the jug (jug_ctrl.v): a leaky accumulator with a
// threshold — but here the accumulator is the neuron MEMBRANE, the input is the
// per-step analog-MAC current (post-ADC, digital), the threshold emits a SPIKE
// (not a weight ±1), and the leak is a real decay (alpha<1) rather than the
// jug's pure integrator (lambda=1).
//
//   v_t   = floor(ALPHA * v_{t-1} / 2^ASHIFT) + in_cur         (leaky integrate)
//   spike = (v_t >= THRESH)                                    (fire)
//   v_t  <- v_t - THRESH   if it fired                         (soft reset)
//
// alpha = ALPHA / 2^ASHIFT. Default 230/256 = 0.8984, the fixed-point image of
// the sim's alpha=0.9. The BIT-FAITHFUL reference (ref/lif_ref.py) uses exactly
// this integer arithmetic; tb_lif_cell.v checks the membrane trace + spikes
// bit-exact against it. (Python `>>` and Verilog `>>>` are both arithmetic /
// floor for two's complement, so they agree.)
//
// Positive-threshold LIF: the membrane may go negative (it only leaks); a spike
// is emitted on the positive crossing only, matching 1[a>=theta] in the paper.
// =============================================================================

module lif_cell #(
    parameter integer MEMW   = 20,   // membrane width (signed)
    parameter integer INW    = 12,   // per-step input current width (signed)
    parameter integer ALPHA  = 230,  // decay numerator; alpha = ALPHA/2^ASHIFT
    parameter integer ASHIFT = 8,
    parameter integer THRESH = 1024, // fire threshold, in membrane LSBs (== float 1.0)
    parameter integer PW     = 8,    // readiness fractional bits (psi in Q_PW; 1.0 == 2^PW)
    parameter integer SURR   = 5,    // surrogate slope (now baked into the LUT; kept for interface)
    parameter integer PSI_N  = 512,  // readiness LUT entries
    parameter integer PSI_IDXSH = 3, // LUT index = |v-THR| >> PSI_IDXSH
    parameter PSI_HEX = "../ref/psi_lut.hex"
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   en,       // advance one timestep
    input  wire                   clr,      // clear membrane (start of sequence)
    input  wire signed [INW-1:0]  in_cur,   // W.s for this neuron, this step
    output reg  signed [MEMW-1:0] mem,      // membrane after the update
    output reg                    spike,    // 1 if it fired this step
    output reg        [PW:0]      psi       // readiness = 1/(1+SURR|v-THR|)^2 (Q_PW), pre-reset membrane
);
    // leak: SIGNED multiply then arithmetic right shift (== floor for two's
    // complement, matching Python `>>`). Keep every intermediate FULL-WIDTH and
    // SIGNED — a part-select (e.g. decayed[MEMW:0]) is unsigned and silently
    // corrupts the sign of a negative membrane. Truncate ONLY at the register.
    wire signed [MEMW+ASHIFT:0] prod    = $signed(mem) * $signed(ALPHA);
    wire signed [MEMW+ASHIFT:0] decayed = prod >>> ASHIFT;
    wire signed [MEMW+ASHIFT:0] integ   = decayed + $signed(in_cur);
    wire                        fire    = (integ >= $signed(THRESH));
    wire signed [MEMW+ASHIFT:0] nxt     = fire ? (integ - $signed(THRESH)) : integ;

    // readiness psi = 1/(1 + SURR*|v-THR|)^2, on the PRE-reset membrane (v-THR == d).
    // ⚡ refinement #3: a SHARED LUT (one ROM for all neurons) replaces the per-neuron divider.
    //   idx = |d| >> PSI_IDXSH (clamped); psi = ROM[idx] (psi -> 0 past the table).
    reg  [PW:0] psi_rom [0:PSI_N-1];
    initial $readmemh(PSI_HEX, psi_rom);
    wire signed [MEMW+ASHIFT:0] d    = integ - $signed(THRESH);
    wire        [MEMW+ASHIFT:0] ad   = d[MEMW+ASHIFT] ? (~d + 1'b1) : d;
    wire        [31:0]          idxr = ad >> PSI_IDXSH;
    wire        [31:0]          idx  = (idxr >= PSI_N) ? (PSI_N - 1) : idxr;
    wire        [PW:0]          psi_c = psi_rom[idx];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem   <= {MEMW{1'b0}};
            spike <= 1'b0;
            psi   <= {(PW+1){1'b0}};
        end else if (clr) begin
            mem   <= {MEMW{1'b0}};
            spike <= 1'b0;
            psi   <= {(PW+1){1'b0}};
        end else if (en) begin
            spike <= fire;
            mem   <= nxt[MEMW-1:0];   // two's-complement truncate to membrane width
            psi   <= psi_c;
        end
    end
endmodule
`default_nettype wire
