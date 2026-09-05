`default_nettype none
`timescale 1ns/1ps
// tb_elig_buffer — bit-exact cosim of elig_buffer.v vs ref/elig_ref.py.
// Vectors from ref/elig_*.hex (regenerate:  python3 ../ref/elig_ref.py).
// Verdict: "ALL PASS" / "FAIL (n mismatches)". Uses !== (x-safe).
module tb_elig_buffer;
    localparam integer DW = 12, N = 8, ACCW = 18, T = 48;

    reg                    clk = 1'b0, rst_n = 1'b0, en = 1'b0, clr = 1'b0;
    reg  signed [DW-1:0]   delta = {DW{1'b0}};
    reg                    spk = 1'b0;
    wire signed [ACCW-1:0] g;

    elig_buffer #(.DW(DW), .N(N), .ACCW(ACCW)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .delta(delta), .spk(spk), .g(g));

    always #5 clk = ~clk;

    reg signed [DW-1:0]   d_v   [0:T-1];
    reg        [0:0]      s_v   [0:T-1];
    reg signed [ACCW-1:0] g_e   [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/elig_delta.hex", d_v);
        $readmemh("../ref/elig_spk.hex",   s_v);
        $readmemh("../ref/elig_g_exp.hex", g_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;    // clear FIFO + accumulator

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            delta = d_v[i];
            spk   = s_v[i][0];
            en    = 1'b1;
            @(posedge clk);            // g now reflects the window ending at step i
            #1;
            en = 1'b0;
            if (g !== g_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d delta=%0d spk=%b | g rtl=%0d exp=%0d",
                         i, d_v[i], s_v[i][0], g, g_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_elig_buffer: ALL PASS (%0d steps, 0 mismatches)", T);
        else           $display("tb_elig_buffer: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
