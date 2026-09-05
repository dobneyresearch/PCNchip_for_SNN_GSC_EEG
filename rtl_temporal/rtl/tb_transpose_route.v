`default_nettype none
`timescale 1ns/1ps
// tb_transpose_route — bit-exact cosim of transpose_route.v vs ref/transpose_ref.py.
// Combinational MAC: apply NV weight/adjoint vectors, check routed errors. Uses !==.
// Vectors: ref/tr_*.hex (regenerate:  python3 ../ref/transpose_ref.py).
module tb_transpose_route;
    localparam integer NO = 2, NI = 2, AWACC = 22, WW = 8, EEW = 12, NV = 24;

    reg  [NO*AWACC-1:0]  delta_flat;
    reg  [NO*NI*WW-1:0]  W_flat;
    wire [NI*EEW-1:0]    e_out_flat;

    transpose_route #(.NO(NO), .NI(NI), .AWACC(AWACC), .WW(WW), .EEW(EEW)) dut (
        .delta_flat(delta_flat), .W_flat(W_flat), .e_out_flat(e_out_flat));

    reg signed [AWACC-1:0] d_v [0:NV*NO-1];
    reg signed [WW-1:0]    w_v [0:NV*NO*NI-1];
    reg signed [EEW-1:0]   e_e [0:NV*NI-1];

    integer k, o, i, mism;
    reg signed [EEW-1:0] e_rtl;
    initial begin
        $readmemh("../ref/tr_delta.hex", d_v);
        $readmemh("../ref/tr_w.hex",     w_v);
        $readmemh("../ref/tr_e_exp.hex", e_e);

        mism = 0;
        for (k = 0; k < NV; k = k + 1) begin
            for (o = 0; o < NO; o = o + 1)
                delta_flat[o*AWACC +: AWACC] = d_v[k*NO + o];
            for (o = 0; o < NO*NI; o = o + 1)
                W_flat[o*WW +: WW] = w_v[k*NO*NI + o];
            #1;
            for (i = 0; i < NI; i = i + 1) begin
                e_rtl = e_out_flat[i*EEW +: EEW];
                if (e_rtl !== e_e[k*NI + i]) begin
                    mism = mism + 1;
                    $display("MISMATCH vec=%0d i=%0d | e rtl=%0d exp=%0d", k, i, e_rtl, e_e[k*NI+i]);
                end
            end
        end

        if (mism == 0) $display("tb_transpose_route: ALL PASS (%0d vectors x %0d outs, 0 mismatches)", NV, NI);
        else           $display("tb_transpose_route: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
