`default_nettype none
`timescale 1ns/1ps
// tb_temporal_2layer — bit-exact cosim of pcn_temporal_2layer.v vs ref/l2_ref.py.
// Deep credit: checks E1,E2,dW1,dW2 each cycle. Vectors: ref/l2_*.hex. Uses !==.
module tb_temporal_2layer;
    localparam integer INW=12, EEW=12, ECW=32, DWW=20, WW=8, N=8, T=40;
    localparam integer NC = T + N + 1;

    reg                   clk=0, rst_n=0, clr=0, en=0, fold=0, z0=0;
    reg  signed [INW-1:0] ic1=0, ic2=0;
    reg  signed [EEW-1:0] e2=0;
    reg  signed [WW-1:0]  w2=0;
    wire                  s1;
    wire signed [ECW-1:0] E1, E2;
    wire signed [DWW-1:0] dW1, dW2;

    pcn_temporal_2layer dut (
        .clk(clk),.rst_n(rst_n),.clr(clr),.en(en),.fold(fold),
        .in_cur1(ic1),.in_cur2(ic2),.e2(e2),.z0(z0),.w2(w2),
        .s1(s1),.E1(E1),.E2(E2),.dW1(dW1),.dW2(dW2));

    always #5 clk = ~clk;

    reg [0:0] en_v[0:NC-1], fd_v[0:NC-1], z_v[0:NC-1];
    reg signed [INW-1:0] i1_v[0:NC-1], i2_v[0:NC-1];
    reg signed [EEW-1:0] e_v[0:NC-1];
    reg signed [WW-1:0]  w_v[0:NC-1];
    reg signed [ECW-1:0] E1e[0:NC-1], E2e[0:NC-1];
    reg signed [DWW-1:0] W1e[0:NC-1], W2e[0:NC-1];

    integer c, mism;
    initial begin
        $readmemh("../ref/l2_en.hex",en_v);   $readmemh("../ref/l2_fold.hex",fd_v);
        $readmemh("../ref/l2_ic1.hex",i1_v);  $readmemh("../ref/l2_ic2.hex",i2_v);
        $readmemh("../ref/l2_e2.hex",e_v);    $readmemh("../ref/l2_z0.hex",z_v);
        $readmemh("../ref/l2_w2.hex",w_v);
        $readmemh("../ref/l2_E1_exp.hex",E1e); $readmemh("../ref/l2_E2_exp.hex",E2e);
        $readmemh("../ref/l2_dW1_exp.hex",W1e);$readmemh("../ref/l2_dW2_exp.hex",W2e);

        @(negedge clk); @(negedge clk); rst_n=1; clr=1; @(negedge clk); clr=0;
        mism=0;
        for (c=0;c<NC;c=c+1) begin
            en=en_v[c][0]; fold=fd_v[c][0]; ic1=i1_v[c]; ic2=i2_v[c];
            e2=e_v[c]; z0=z_v[c][0]; w2=w_v[c];
            @(posedge clk); #1; en=0; fold=0;
            if (E1!==E1e[c]||E2!==E2e[c]||dW1!==W1e[c]||dW2!==W2e[c]) begin
                mism=mism+1;
                $display("MISMATCH c=%0d | E1 %0d/%0d E2 %0d/%0d dW1 %0d/%0d dW2 %0d/%0d",
                         c,E1,E1e[c],E2,E2e[c],dW1,W1e[c],dW2,W2e[c]);
            end
            @(negedge clk);
        end
        if (mism==0) $display("tb_temporal_2layer: ALL PASS (%0d cycles, 0 mismatches)", NC);
        else         $display("tb_temporal_2layer: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
