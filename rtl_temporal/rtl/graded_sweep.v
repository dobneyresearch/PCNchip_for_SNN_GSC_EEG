`default_nettype none
// =============================================================================
// graded_sweep — the graded write with ONE shared sqrt+divide, swept (refinement #4).
// =============================================================================
// Instead of a graded_write (with its own √+÷) per synapse, hold the per-synapse
// (m,v) state in an array and time-multiplex a SINGLE step-compute engine across
// the synapses — one per cycle — exactly as jug_ctrl sweeps one shared comparator.
// The per-synapse arithmetic is identical to graded_write (already verified), so the
// swept result is bit-exact to NS parallel graded_writes; the win is area (one
// √+÷, not NS). A fold takes NS cycles. Bit-faithful ref: ref/gsweep_ref.py.
// =============================================================================

module graded_sweep #(
    parameter integer NS   = 4,    // synapses swept by the one engine
    parameter integer EW   = 16, parameter integer MW = 24, parameter integer VW = 32,
    parameter integer B1   = 230, parameter integer BSH1 = 8,
    parameter integer B2   = 1014, parameter integer BSH2 = 10,
    parameter integer F    = 10, parameter integer CLIP = 1024, parameter integer EPS = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clr,
    input  wire                     fold_start,   // begin a sweep over the NS synapses
    input  wire [NS*EW-1:0]         E_flat,       // per-synapse folded credit (latched)
    output reg                      busy,
    output reg                      done,         // 1-cycle pulse when the sweep finishes
    output wire [NS*(F+4)-1:0]      dW_flat       // per-synapse code step
);
    // ── the ONE shared floor integer sqrt (== graded_write / Python math.isqrt) ──
    function automatic [15:0] isqrt32;
        input [31:0] x;
        reg [31:0] num, bitv, res;
        begin
            num = x; res = 0; bitv = 32'h4000_0000;
            while (bitv > num) bitv = bitv >> 2;
            while (bitv != 0) begin
                if (num >= res + bitv) begin num = num - (res + bitv); res = (res >> 1) + bitv; end
                else res = res >> 1;
                bitv = bitv >> 2;
            end
            isqrt32 = res[15:0];
        end
    endfunction

    reg signed [MW-1:0] m  [0:NS-1];
    reg        [VW-1:0] v  [0:NS-1];
    reg signed [F+3:0]  dW [0:NS-1];
    reg signed [EW-1:0] Er [0:NS-1];               // latched E for the sweep
    integer k;
    reg [31:0] cnt;
    reg        state;                              // 0 IDLE, 1 SWEEP

    // ── shared step-compute for the CURRENT synapse `cnt` (one √+÷ instance) ──
    wire signed [MW-1:0]      mc = m[cnt];
    wire        [VW-1:0]      vc = v[cnt];
    wire signed [EW-1:0]      Ec = Er[cnt];
    wire signed [MW+BSH1+9:0] m_acc = $signed(B1)*$signed(mc) + $signed((1<<BSH1)-B1)*$signed(Ec);
    wire signed [MW-1:0]      m_nxt = m_acc >>> BSH1;
    wire signed [2*EW-1:0]    Ewd   = $signed(Ec);
    wire        [2*EW-1:0]    e2    = $unsigned(Ewd * Ewd);
    wire        [63:0]        v_acc = B2*vc + ((1<<BSH2)-B2)*e2;
    wire        [VW-1:0]      v_nxt = v_acc >> BSH2;
    wire        [VW-1:0]      v_in  = v_nxt + EPS[VW-1:0];
    wire        [15:0]        vs    = isqrt32(v_in[31:0]);
    wire        [15:0]        denom = (vs == 16'd0) ? 16'd1 : vs;
    wire        [MW-1:0]      m_abs = m_nxt[MW-1] ? (~m_nxt + 1'b1) : m_nxt;
    wire        [MW+F-1:0]    num   = m_abs * (1 << F);
    wire        [MW+F-1:0]    q_raw = num / {{(MW+F-16){1'b0}}, denom};
    localparam signed [MW+F-1:0] CLIPW = CLIP;
    wire        [MW+F-1:0]    q_clip = (q_raw > CLIPW) ? CLIPW : q_raw;
    wire signed [F+3:0]       q_s   = $signed({1'b0, q_clip[F+1:0]});
    wire signed [F+3:0]       step_c = m_nxt[MW-1] ? -q_s : q_s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || clr) begin
            state <= 1'b0; busy <= 1'b0; done <= 1'b0; cnt <= 32'd0;
            for (k = 0; k < NS; k = k + 1) begin m[k] <= 0; v[k] <= 0; dW[k] <= 0; end
        end else begin
            done <= 1'b0;
            case (state)
                1'b0: if (fold_start) begin
                    for (k = 0; k < NS; k = k + 1) Er[k] <= E_flat[k*EW +: EW];
                    cnt <= 32'd0; busy <= 1'b1; state <= 1'b1;
                end
                1'b1: begin
                    m[cnt]  <= m_nxt;
                    v[cnt]  <= v_nxt;
                    dW[cnt] <= step_c;
                    if (cnt == NS-1) begin busy <= 1'b0; done <= 1'b1; state <= 1'b0; end
                    else cnt <= cnt + 32'd1;
                end
            endcase
        end
    end

    genvar g;
    generate for (g = 0; g < NS; g = g + 1)
        assign dW_flat[g*(F+4) +: (F+4)] = dW[g];
    endgenerate
endmodule
`default_nettype wire
