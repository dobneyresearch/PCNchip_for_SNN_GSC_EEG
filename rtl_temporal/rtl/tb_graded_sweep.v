`default_nettype none
`timescale 1ns/1ps
// tb_graded_sweep — bit-exact cosim of graded_sweep.v vs ref/gsweep_ref.py.
// Drives NFOLDS folds; each pulses fold_start, waits for done, checks all NS dW. Uses !==.
// Vectors: ref/gs_*.hex (regenerate:  python3 ../ref/gsweep_ref.py).
module tb_graded_sweep;
    localparam integer NS = 4, EW = 16, F = 10, NFOLDS = 12;
    localparam integer SW = F + 4;

    reg                   clk = 1'b0, rst_n = 1'b0, clr = 1'b0, fold_start = 1'b0;
    reg  [NS*EW-1:0]      E_flat = {(NS*EW){1'b0}};
    wire                  busy, done;
    wire [NS*SW-1:0]      dW_flat;

    graded_sweep #(.NS(NS), .EW(EW), .F(F)) dut (
        .clk(clk), .rst_n(rst_n), .clr(clr), .fold_start(fold_start),
        .E_flat(E_flat), .busy(busy), .done(done), .dW_flat(dW_flat));

    always #5 clk = ~clk;

    reg signed [EW-1:0] E_v  [0:NFOLDS*NS-1];
    reg signed [SW-1:0] dW_e [0:NFOLDS*NS-1];

    integer f, s, mism, guard;
    reg signed [SW-1:0] w_rtl;
    initial begin
        $readmemh("../ref/gs_E.hex",      E_v);
        $readmemh("../ref/gs_dW_exp.hex", dW_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1; @(negedge clk);

        mism = 0;
        for (f = 0; f < NFOLDS; f = f + 1) begin
            for (s = 0; s < NS; s = s + 1)
                E_flat[s*EW +: EW] = E_v[f*NS + s];
            fold_start = 1'b1; @(negedge clk); fold_start = 1'b0;
            // wait for the sweep to finish
            guard = 0;
            while (!done && guard < 100) begin @(negedge clk); guard = guard + 1; end
            #1;
            for (s = 0; s < NS; s = s + 1) begin
                w_rtl = dW_flat[s*SW +: SW];
                if (w_rtl !== dW_e[f*NS + s]) begin
                    mism = mism + 1;
                    $display("MISMATCH fold=%0d syn=%0d | dW rtl=%0d exp=%0d",
                             f, s, w_rtl, dW_e[f*NS + s]);
                end
            end
        end

        if (mism == 0) $display("tb_graded_sweep: ALL PASS (%0d folds x %0d synapses, 0 mismatches)", NFOLDS, NS);
        else           $display("tb_graded_sweep: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
