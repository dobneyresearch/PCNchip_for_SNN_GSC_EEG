`default_nettype none
// =============================================================================
// pcn_temporal_2layer — DEEP forwards-only credit (Paper III), 1 neuron/layer.
// =============================================================================
// The top learning signal trains BOTH layers with no backward pass: layer-2's
// adjoint δ2 both updates W2 AND routes down through the Wᵀ hop to become layer-1's
// error, which drives layer-1's own adjoint and update.
//
//   layer2:  lif2 -> ψ2 ; a2=ψ2·e2 ; adjoint2 -> δ2 ; E2 += z1_del·δ2   (z1 = s1)
//   route :  e1 = Wᵀ δ2                                   (transpose_route)
//   layer1:  lif1 -> ψ1 ; a1=ψ1·e1 ; adjoint1 -> δ1 ; E1 += z0_del·δ1   (z0 = input)
//   fold  :  graded_write per layer -> dW2, dW1
//
// Forward currents in_cur1/in_cur2 are provided (the analog MAC is out of scope);
// the demonstration is the CREDIT routing. Bit-faithful ref: ref/l2_ref.py.
// =============================================================================

module pcn_temporal_2layer #(
    parameter integer INW = 12, parameter integer EEW = 12,
    parameter integer MEMW = 20, parameter integer ALPHA = 230, parameter integer ASHIFT = 8,
    parameter integer THRESH = 1024, parameter integer PW = 8, parameter integer SURR = 5,
    parameter integer DW = 16, parameter integer N = 8, parameter integer AWACC = 22,
    parameter integer ECW = 32, parameter integer GEW = 16, parameter integer DWW = 20,
    parameter integer WW = 8, parameter integer RSH = 16
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clr,
    input  wire                   en,
    input  wire                   fold,
    input  wire signed [INW-1:0]  in_cur1,   // layer-1 forward current
    input  wire signed [INW-1:0]  in_cur2,   // layer-2 forward current
    input  wire signed [EEW-1:0]  e2,        // top error (into layer 2)
    input  wire                   z0,        // layer-1 presyn spike (network input)
    input  wire signed [WW-1:0]   w2,        // layer-2 weight (for the Wᵀ hop)
    output wire                   s1,         // layer-1 spike (= layer-2 presyn)
    output reg  signed [ECW-1:0]  E1,
    output reg  signed [ECW-1:0]  E2,
    output wire signed [DWW-1:0]  dW1,
    output wire signed [DWW-1:0]  dW2
);
    wire signed [MEMW-1:0] mem1, mem2;
    wire s2; wire [PW:0] psi1, psi2;

    lif_cell #(.MEMW(MEMW),.INW(INW),.ALPHA(ALPHA),.ASHIFT(ASHIFT),.THRESH(THRESH),.PW(PW),.SURR(SURR))
        u_lif1 (.clk(clk),.rst_n(rst_n),.en(en),.clr(clr),.in_cur(in_cur1),.mem(mem1),.spike(s1),.psi(psi1));
    lif_cell #(.MEMW(MEMW),.INW(INW),.ALPHA(ALPHA),.ASHIFT(ASHIFT),.THRESH(THRESH),.PW(PW),.SURR(SURR))
        u_lif2 (.clk(clk),.rst_n(rst_n),.en(en),.clr(clr),.in_cur(in_cur2),.mem(mem2),.spike(s2),.psi(psi2));

    // layer 2 adjoint
    wire signed [PW+1+EEW:0] pe2 = $signed({1'b0,psi2}) * $signed(e2);
    wire signed [DW-1:0] a2 = (pe2 >>> PW);
    wire signed [AWACC-1:0] delta2;
    adjoint_window #(.DW(DW),.N(N),.ACCW(AWACC)) u_adj2
        (.clk(clk),.rst_n(rst_n),.en(en),.clr(clr),.a(a2),.delta(delta2));

    // Wᵀ hop: e1_raw = transpose(delta2, w2)
    wire [EEW-1:0] e1_flat;
    transpose_route #(.NO(1),.NI(1),.AWACC(AWACC),.WW(WW),.EEW(EEW),.RSH(RSH)) u_tr
        (.delta_flat(delta2), .W_flat(w2), .e_out_flat(e1_flat));
    wire signed [EEW-1:0] e1_raw = e1_flat;

    // router-fit #1: quantise the routed adjoint to the 6-bit interconnect message,
    // then reconstruct the error the receiver (layer 1) uses: e1 = q*scale/L.
    wire signed [5:0]    rq_q;
    wire [EEW-1:0]       rq_scale;
    route_quant #(.VW(EEW), .OUTB(6), .LEAK(6)) u_rq (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .value(e1_raw), .q(rq_q), .scale(rq_scale));
    localparam integer L6 = 31;
    wire [4:0]        rq_qm = rq_q[5] ? (~rq_q + 1'b1) : rq_q;         // |q|
    wire [EEW+5:0]    recon = (rq_qm * rq_scale) / L6;                 // |q|*scale/L
    wire signed [EEW-1:0] e1 = rq_q[5] ? -$signed({1'b0, recon[EEW-2:0]})
                                       :  $signed({1'b0, recon[EEW-2:0]});

    // layer 1 adjoint (on the 6-bit-quantised, reconstructed error)
    wire signed [PW+1+EEW:0] pe1 = $signed({1'b0,psi1}) * $signed(e1);
    wire signed [DW-1:0] a1 = (pe1 >>> PW);
    wire signed [AWACC-1:0] delta1;
    adjoint_window #(.DW(DW),.N(N),.ACCW(AWACC)) u_adj1
        (.clk(clk),.rst_n(rst_n),.en(en),.clr(clr),.a(a1),.delta(delta1));

    // presyn delays: z0 (network input) for layer1; s1 (layer1 spike) for layer2
    reg [N-1:0] z0sr, z1sr;
    wire z0_del = z0sr[N-1];
    wire z1_del = z1sr[N-1];

    wire signed [ECW-1:0] d1x = {{(ECW-AWACC){delta1[AWACC-1]}}, delta1};
    wire signed [ECW-1:0] d2x = {{(ECW-AWACC){delta2[AWACC-1]}}, delta2};

    localparam signed [ECW-1:0] SP = (1<<<(GEW-1))-1, SN = -(1<<<(GEW-1));
    wire signed [GEW-1:0] ef1 = (E1>SP)?SP[GEW-1:0]:(E1<SN)?SN[GEW-1:0]:E1[GEW-1:0];
    wire signed [GEW-1:0] ef2 = (E2>SP)?SP[GEW-1:0]:(E2<SN)?SN[GEW-1:0]:E2[GEW-1:0];
    graded_write #(.EW(GEW)) u_gw1 (.clk(clk),.rst_n(rst_n),.en(fold),.clr(clr),.E(ef1),.m(),.v(),.step(dW1));
    graded_write #(.EW(GEW)) u_gw2 (.clk(clk),.rst_n(rst_n),.en(fold),.clr(clr),.E(ef2),.m(),.v(),.step(dW2));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            E1 <= {ECW{1'b0}}; E2 <= {ECW{1'b0}}; z0sr <= {N{1'b0}}; z1sr <= {N{1'b0}};
        end else if (clr) begin
            E1 <= {ECW{1'b0}}; E2 <= {ECW{1'b0}}; z0sr <= {N{1'b0}}; z1sr <= {N{1'b0}};
        end else if (en) begin
            z0sr <= {z0sr[N-2:0], z0};
            z1sr <= {z1sr[N-2:0], s1};        // layer-1 spike feeds layer-2 presyn
            if (z0_del) E1 <= E1 + d1x;
            if (z1_del) E2 <= E2 + d2x;
        end else if (fold) begin
            E1 <= {ECW{1'b0}}; E2 <= {ECW{1'b0}};
        end
    end
endmodule
`default_nettype wire
