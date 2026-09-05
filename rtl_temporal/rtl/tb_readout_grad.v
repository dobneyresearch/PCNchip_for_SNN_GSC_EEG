`default_nettype none
`timescale 1ns/1ps
// tb_readout_grad — bit-exact cosim of readout_grad.v vs ref/readout_ref.py.
// Replays the op stream (1=acc, 2=fold). Checks E every cycle and dW every cycle.
// Vectors: ref/ro_*.hex (regenerate:  python3 ../ref/readout_ref.py). Uses !==.
module tb_readout_grad;
    localparam integer EW = 12, FW = 12, ACCW = 32, LR_MULT = 100, LSH = 16, CLIP = 127, T = 40;

    reg                   clk = 1'b0, rst_n = 1'b0, acc = 1'b0, fold = 1'b0, bias_mode = 1'b0;
    reg  signed [EW-1:0]  e = {EW{1'b0}};
    reg  signed [FW-1:0]  f = {FW{1'b0}};
    wire signed [ACCW-1:0] E;
    wire signed [8:0]     dW;

    readout_grad #(.EW(EW), .FW(FW), .ACCW(ACCW), .LR_MULT(LR_MULT), .LSH(LSH), .CLIP(CLIP)) dut (
        .clk(clk), .rst_n(rst_n), .acc(acc), .fold(fold), .bias_mode(bias_mode),
        .e(e), .f(f), .E(E), .dW(dW));

    always #5 clk = ~clk;

    reg [3:0]            op_v  [0:T-1];
    reg signed [EW-1:0]  e_v   [0:T-1];
    reg signed [FW-1:0]  f_v   [0:T-1];
    reg [0:0]            b_v   [0:T-1];
    reg signed [ACCW-1:0] E_e  [0:T-1];
    reg signed [8:0]     dW_e  [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/ro_op.hex",    op_v);
        $readmemh("../ref/ro_e.hex",     e_v);
        $readmemh("../ref/ro_f.hex",     f_v);
        $readmemh("../ref/ro_bias.hex",  b_v);
        $readmemh("../ref/ro_E_exp.hex", E_e);
        $readmemh("../ref/ro_dW_exp.hex",dW_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1; @(negedge clk);

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            e         = e_v[i];
            f         = f_v[i];
            bias_mode = b_v[i][0];
            acc       = (op_v[i] == 4'd1);
            fold      = (op_v[i] == 4'd2);
            @(posedge clk);
            #1;
            acc = 1'b0; fold = 1'b0;
            if (E !== E_e[i] || dW !== dW_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d op=%0d e=%0d f=%0d bias=%b | E %0d/%0d dW %0d/%0d",
                         i, op_v[i], e_v[i], f_v[i], b_v[i][0], E, E_e[i], dW, dW_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_readout_grad: ALL PASS (%0d cycles, 0 mismatches)", T);
        else           $display("tb_readout_grad: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
