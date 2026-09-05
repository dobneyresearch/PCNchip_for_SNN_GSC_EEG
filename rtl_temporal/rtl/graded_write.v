`default_nettype none
// =============================================================================
// graded_write — reliability-graded weight step m/sqrt(v) (Paper III eq. 5).
// =============================================================================
// The reliability-grading role of the leaky-accumulator primitive (var_norm).
// Two leaky-accumulator EMAs of the folded credit E — its mean m and its energy
// v — produce a self-scaling, bounded weight step:
//
//   m = (B1*m + (2^BSH1 - B1)*E)   >>> BSH1        (signed;  beta1 = B1/2^BSH1 ~ 0.9)
//   v = (B2*v + (2^BSH2 - B2)*E^2) >>  BSH2        (unsigned; beta2 = B2/2^BSH2 ~ 0.99)
//   step = sign(m) * clamp( (|m| << F) / isqrt(v+EPS), CLIP )
//
// Large when the evidence is consistent (m large, v moderate); small when it is
// noisy (m ~ 0, v large) — the mechanism that is load-bearing on TEMPORAL credit
// (a noisy per-timestep sum) and a near-no-op on aggregate/rate credit.
//
// The reciprocal-root is a floor integer-sqrt + divide here (fully bit-definable,
// so the Python reference matches exactly). A normalise+LUT rsqrt is the area-
// optimal silicon alternative; the sqrt/divide would be pipelined, not 1-cycle.
// Bit-faithful reference: ref/graded_ref.py.  tb_graded_write.v checks m,v,step.
// =============================================================================

module graded_write #(
    parameter integer EW   = 16,   // folded-credit width (signed)
    parameter integer MW   = 24,   // mean EMA width (signed)
    parameter integer VW   = 32,   // energy EMA width (unsigned)
    parameter integer B1   = 230,  // beta1 numerator
    parameter integer BSH1 = 8,
    parameter integer B2   = 1014, // beta2 numerator
    parameter integer BSH2 = 10,
    parameter integer F    = 10,   // step fractional bits (Q_F, so 1.0 == 2^F)
    parameter integer CLIP = 1024, // step clamp (== 1.0 in Q_F)
    parameter integer EPS  = 1
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,     // one fold event
    input  wire                 clr,
    input  wire signed [EW-1:0] E,      // folded credit for this synapse
    output reg  signed [MW-1:0] m,      // mean EMA (checked)
    output reg        [VW-1:0]  v,      // energy EMA (checked)
    output reg  signed [F+3:0]  step    // graded weight step (Q_F, clamped)
);
    // ── floor integer sqrt (digit-by-digit); matches Python math.isqrt ──
    function automatic [15:0] isqrt32;
        input [31:0] x;
        reg [31:0] num, bitv, res;
        begin
            num = x; res = 0; bitv = 32'h4000_0000;   // 4^15
            while (bitv > num) bitv = bitv >> 2;
            while (bitv != 0) begin
                if (num >= res + bitv) begin
                    num = num - (res + bitv);
                    res = (res >> 1) + bitv;
                end else begin
                    res = res >> 1;
                end
                bitv = bitv >> 2;
            end
            isqrt32 = res[15:0];
        end
    endfunction

    // ── m EMA (signed) ──
    wire signed [MW+BSH1+9:0] m_acc  = $signed(B1) * $signed(m)
                                     + $signed((1 << BSH1) - B1) * $signed(E);
    wire signed [MW+BSH1+9:0] m_shr  = m_acc >>> BSH1;
    wire signed [MW-1:0]      m_nxt  = m_shr[MW-1:0];

    // ── v EMA (unsigned) ──
    // ⚠ widen BEFORE squaring: $signed(E)*$signed(E) inside $unsigned() is a
    //   SELF-DETERMINED 16-bit multiply and would truncate E^2 to 16 bits.
    wire signed [2*EW-1:0] Ew = $signed(E);                       // sign-extend to 2*EW
    wire [2*EW-1:0] e2      = $unsigned(Ew * Ew);                 // E^2 >= 0, full width
    wire [63:0]     v_acc   = B2 * v + ((1 << BSH2) - B2) * e2;
    wire [63:0]     v_shr   = v_acc >> BSH2;
    wire [VW-1:0]   v_nxt   = v_shr[VW-1:0];

    // ── step = sign(m) * clamp(|m|<<F / isqrt(v+EPS), CLIP) ──
    wire [VW-1:0]     v_in   = v_nxt + EPS[VW-1:0];
    wire [15:0]       vs     = isqrt32(v_in[31:0]);
    wire [15:0]       denom  = (vs == 16'd0) ? 16'd1 : vs;
    wire [MW-1:0]     m_abs  = m_nxt[MW-1] ? (~m_nxt + 1'b1) : m_nxt;
    wire [MW+F-1:0]   num    = m_abs * (1 << F);
    wire [MW+F-1:0]   q_raw  = num / {{(MW+F-16){1'b0}}, denom};
    // NB: size CLIP into a full-width local — a bit-select CLIP[MW+F-1:0] is OUT OF
    // RANGE for the 32-bit parameter (bits 32,33 read as x and poison the clamp).
    localparam [MW+F-1:0] CLIPW = CLIP;
    wire [MW+F-1:0]   q_clip = (q_raw > CLIPW) ? CLIPW : q_raw;
    wire signed [F+3:0] q_s  = $signed({1'b0, q_clip[F+1:0]});
    wire signed [F+3:0] step_c = m_nxt[MW-1] ? -q_s : q_s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m <= {MW{1'b0}}; v <= {VW{1'b0}}; step <= {(F+4){1'b0}};
        end else if (clr) begin
            m <= {MW{1'b0}}; v <= {VW{1'b0}}; step <= {(F+4){1'b0}};
        end else if (en) begin
            m <= m_nxt; v <= v_nxt; step <= step_c;
        end
    end
endmodule
`default_nettype wire
