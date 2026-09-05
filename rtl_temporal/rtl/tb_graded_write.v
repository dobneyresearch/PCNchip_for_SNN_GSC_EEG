`default_nettype none
`timescale 1ns/1ps
// tb_graded_write — bit-exact cosim of graded_write.v vs ref/graded_ref.py.
// Vectors from ref/graded_*.hex (regenerate:  python3 ../ref/graded_ref.py).
// Checks m, v AND step bit-exact. Verdict "ALL PASS" / "FAIL (n)". Uses !==.
module tb_graded_write;
    localparam integer EW = 16, MW = 24, VW = 32,
                       B1 = 230, BSH1 = 8, B2 = 1014, BSH2 = 10,
                       F = 10, CLIP = 1024, EPS = 1, T = 40;

    reg                  clk = 1'b0, rst_n = 1'b0, en = 1'b0, clr = 1'b0;
    reg  signed [EW-1:0] E = {EW{1'b0}};
    wire signed [MW-1:0] m;
    wire        [VW-1:0] v;
    wire signed [F+3:0]  step;

    graded_write #(.EW(EW), .MW(MW), .VW(VW), .B1(B1), .BSH1(BSH1),
                   .B2(B2), .BSH2(BSH2), .F(F), .CLIP(CLIP), .EPS(EPS)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .E(E),
        .m(m), .v(v), .step(step));

    always #5 clk = ~clk;

    reg signed [EW-1:0]  E_v    [0:T-1];
    reg signed [MW-1:0]  m_e    [0:T-1];
    reg        [VW-1:0]  v_e    [0:T-1];
    reg signed [F+3:0]   step_e [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/graded_E.hex",        E_v);
        $readmemh("../ref/graded_m_exp.hex",    m_e);
        $readmemh("../ref/graded_v_exp.hex",    v_e);
        $readmemh("../ref/graded_step_exp.hex", step_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            E  = E_v[i];
            en = 1'b1;
            @(posedge clk);           // m,v,step now reflect fold i
            #1;
            en = 1'b0;
            if (m !== m_e[i] || v !== v_e[i] || step !== step_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d E=%0d | m %0d/%0d v %0d/%0d step %0d/%0d",
                         i, E_v[i], m, m_e[i], v, v_e[i], step, step_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_graded_write: ALL PASS (%0d folds, 0 mismatches)", T);
        else           $display("tb_graded_write: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
