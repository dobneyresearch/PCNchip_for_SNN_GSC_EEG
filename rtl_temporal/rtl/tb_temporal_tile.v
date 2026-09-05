`default_nettype none
`timescale 1ns/1ps
// tb_temporal_tile — bit-exact cosim of pcn_temporal_tile.v vs ref/tile_ref.py.
// NO x NI synapse tile. Checks every synapse's E and dW each cycle. Uses !==.
// Vectors: ref/tile_*.hex (regenerate:  python3 ../ref/tile_ref.py).
module tb_temporal_tile;
    localparam integer NO = 2, NI = 2, INW = 12, EEW = 12, ECW = 32, DWW = 20, N = 8, T = 32;
    localparam integer NSYN = NO*NI, NC = T + N + 1;

    reg                        clk = 1'b0, rst_n = 1'b0, clr = 1'b0, en = 1'b0, fold = 1'b0;
    reg  [NO*INW-1:0]          in_flat = {(NO*INW){1'b0}};
    reg  [NO*EEW-1:0]          e_flat  = {(NO*EEW){1'b0}};
    reg  [NI-1:0]              z_pre = {NI{1'b0}};
    wire [NO*NI*ECW-1:0]       E_flat;
    wire [NO*NI*DWW-1:0]       dW_flat;

    pcn_temporal_tile #(.NO(NO), .NI(NI)) dut (
        .clk(clk), .rst_n(rst_n), .clr(clr), .en(en), .fold(fold),
        .in_cur_flat(in_flat), .e_err_flat(e_flat), .z_pre(z_pre),
        .E_flat(E_flat), .dW_flat(dW_flat));

    always #5 clk = ~clk;

    reg [0:0]            en_v [0:NC-1];
    reg [0:0]            fd_v [0:NC-1];
    reg signed [INW-1:0] in_v [0:NC*NO-1];
    reg signed [EEW-1:0] e_v  [0:NC*NO-1];
    reg [NI-1:0]         z_v  [0:NC-1];
    reg signed [ECW-1:0] E_e  [0:NC*NSYN-1];
    reg signed [DWW-1:0] dW_e [0:NC*NSYN-1];

    integer c, s, mism;
    reg signed [ECW-1:0] e_rtl; reg signed [DWW-1:0] w_rtl;
    initial begin
        $readmemh("../ref/tile_en.hex",     en_v);
        $readmemh("../ref/tile_fold.hex",   fd_v);
        $readmemh("../ref/tile_in.hex",     in_v);
        $readmemh("../ref/tile_e.hex",      e_v);
        $readmemh("../ref/tile_z.hex",      z_v);
        $readmemh("../ref/tile_E_exp.hex",  E_e);
        $readmemh("../ref/tile_dW_exp.hex", dW_e);

        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (c = 0; c < NC; c = c + 1) begin
            en = en_v[c][0]; fold = fd_v[c][0]; z_pre = z_v[c];
            for (s = 0; s < NO; s = s + 1) begin
                in_flat[s*INW +: INW] = in_v[c*NO + s];
                e_flat[s*EEW +: EEW]  = e_v[c*NO + s];
            end
            @(posedge clk);
            #1;
            en = 1'b0; fold = 1'b0;
            for (s = 0; s < NSYN; s = s + 1) begin
                e_rtl = E_flat[s*ECW +: ECW];
                w_rtl = dW_flat[s*DWW +: DWW];
                if (e_rtl !== E_e[c*NSYN + s] || w_rtl !== dW_e[c*NSYN + s]) begin
                    mism = mism + 1;
                    $display("MISMATCH cyc=%0d syn=%0d | E %0d/%0d dW %0d/%0d",
                             c, s, e_rtl, E_e[c*NSYN+s], w_rtl, dW_e[c*NSYN+s]);
                end
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_temporal_tile: ALL PASS (%0d cycles x %0d synapses, 0 mismatches)", NC, NSYN);
        else           $display("tb_temporal_tile: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
