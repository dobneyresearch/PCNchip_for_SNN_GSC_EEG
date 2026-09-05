`default_nettype none
`timescale 1ns/1ps
// tb_adjoint_window — bit-exact cosim of adjoint_window.v vs ref/adjoint_ref.py.
// Vectors: ref/adj_*.hex (regenerate:  python3 ../ref/adjoint_ref.py). Uses !==.
module tb_adjoint_window;
    localparam integer DW = 16, N = 8, ACCW = 22, T = 48;

    reg                    clk = 1'b0, rst_n = 1'b0, en = 1'b0, clr = 1'b0;
    reg  signed [DW-1:0]   a = {DW{1'b0}};
    wire signed [ACCW-1:0] delta;

    adjoint_window #(.DW(DW), .N(N), .ACCW(ACCW)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .a(a), .delta(delta));

    always #5 clk = ~clk;

    reg signed [DW-1:0]   a_v [0:T-1];
    reg signed [ACCW-1:0] d_e [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/adj_a.hex",         a_v);
        $readmemh("../ref/adj_delta_exp.hex", d_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            a  = a_v[i];
            en = 1'b1;
            @(posedge clk);
            #1;
            en = 1'b0;
            if (delta !== d_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d a=%0d | delta rtl=%0d exp=%0d", i, a_v[i], delta, d_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_adjoint_window: ALL PASS (%0d steps, 0 mismatches)", T);
        else           $display("tb_adjoint_window: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
