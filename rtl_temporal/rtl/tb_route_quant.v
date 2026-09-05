`default_nettype none
`timescale 1ns/1ps
// tb_route_quant — bit-exact cosim of route_quant.v vs ref/routeq_ref.py.
// Checks the 6-bit message q AND the running scale each cycle. Vectors: ref/rq_*.hex. Uses !==.
module tb_route_quant;
    localparam integer VW = 24, OUTB = 6, LEAK = 6, T = 48;

    reg                    clk = 1'b0, rst_n = 1'b0, en = 1'b0, clr = 1'b0;
    reg  signed [VW-1:0]   value = {VW{1'b0}};
    wire signed [OUTB-1:0] q;
    wire        [VW-1:0]   scale;

    route_quant #(.VW(VW), .OUTB(OUTB), .LEAK(LEAK)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .value(value), .q(q), .scale(scale));

    always #5 clk = ~clk;

    reg signed [VW-1:0]   in_v [0:T-1];
    reg signed [OUTB-1:0] q_e  [0:T-1];
    reg        [VW-1:0]   s_e  [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/rq_in.hex",       in_v);
        $readmemh("../ref/rq_q_exp.hex",    q_e);
        $readmemh("../ref/rq_scale_exp.hex",s_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            value = in_v[i];
            en    = 1'b1;
            @(posedge clk);
            #1;
            en = 1'b0;
            if (q !== q_e[i] || scale !== s_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d value=%0d | q %0d/%0d scale %0d/%0d",
                         i, in_v[i], q, q_e[i], scale, s_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_route_quant: ALL PASS (%0d steps, 0 mismatches)", T);
        else           $display("tb_route_quant: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
