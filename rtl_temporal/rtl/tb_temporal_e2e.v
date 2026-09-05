`default_nettype none
`timescale 1ns/1ps
// =============================================================================
// tb_temporal_e2e — the RTL temporal top replaying a REAL GSC lane.
// =============================================================================
// The end-to-end closing link: the per-module TBs prove RTL == its integer ref,
// and tb_temporal_top proves the lane composes on synthetic random vectors. THIS
// TB drives pcn_temporal_top with a stimulus extracted from the VALIDATED SIM
// (pcn_learner_snn.py) on real GSC data — the exact per-timestep in_cur (membrane
// current Zᵀ·W[o]), e_err (routed top error d_o·g(t)) and presyn spike z_i of one
// top-hidden-layer synapse — quantised to the fixed-point contract. The expected
// E_credit/dW come from ref/e2e_ref.py via top_ref.Top (bit-exact to this RTL), so
// a PASS proves the silicon datapath reproduces the sim's learning signal for a
// real lane, cycle by cycle. Vectors: ref/e2e_*.hex (regenerate: python3
// ../ref/e2e_ref.py). Checks with !== (x-safe). Identical harness to tb_temporal_top.
// =============================================================================
module tb_temporal_e2e;
    localparam integer INW = 12, EEW = 12, ECW = 32, GEW = 16, N = 8, T = 200;
    localparam integer NC = T + N + 1;                 // real + flush + fold

    reg                   clk = 1'b0, rst_n = 1'b0, clr = 1'b0, en = 1'b0, fold = 1'b0, z = 1'b0;
    reg  signed [INW-1:0] in_cur = {INW{1'b0}};
    reg  signed [EEW-1:0] e_err  = {EEW{1'b0}};
    wire                  spike;
    wire [8:0]            psi;
    wire signed [ECW-1:0] E_credit;
    wire signed [19:0]    dW;

    pcn_temporal_top dut (
        .clk(clk), .rst_n(rst_n), .clr(clr), .en(en),
        .in_cur(in_cur), .e_err(e_err), .z_pre(z), .fold(fold),
        .spike(spike), .psi(psi), .E_credit(E_credit), .dW(dW));

    always #5 clk = ~clk;

    reg [0:0]            en_v [0:NC-1];
    reg [0:0]            fd_v [0:NC-1];
    reg signed [INW-1:0] in_v [0:NC-1];
    reg signed [EEW-1:0] e_v  [0:NC-1];
    reg [0:0]            z_v  [0:NC-1];
    reg signed [ECW-1:0] E_e  [0:NC-1];
    reg signed [19:0]    dW_e [0:NC-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/e2e_en.hex",     en_v);
        $readmemh("../ref/e2e_fold.hex",   fd_v);
        $readmemh("../ref/e2e_in.hex",     in_v);
        $readmemh("../ref/e2e_e.hex",      e_v);
        $readmemh("../ref/e2e_z.hex",      z_v);
        $readmemh("../ref/e2e_E_exp.hex",  E_e);
        $readmemh("../ref/e2e_dW_exp.hex", dW_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (i = 0; i < NC; i = i + 1) begin
            en     = en_v[i][0];
            fold   = fd_v[i][0];
            in_cur = in_v[i];
            e_err  = e_v[i];
            z      = z_v[i][0];
            @(posedge clk);
            #1;
            en = 1'b0; fold = 1'b0;
            if (E_credit !== E_e[i] || dW !== dW_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH cyc=%0d en=%b fd=%b | E %0d/%0d dW %0d/%0d",
                         i, en_v[i][0], fd_v[i][0], E_credit, E_e[i], dW, dW_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_temporal_e2e: ALL PASS (%0d cycles real-GSC lane, 0 mismatches)", NC);
        else           $display("tb_temporal_e2e: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
