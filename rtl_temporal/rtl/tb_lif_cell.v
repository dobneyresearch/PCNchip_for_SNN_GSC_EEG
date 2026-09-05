`default_nettype none
`timescale 1ns/1ps
// tb_lif_cell — bit-exact cosim of lif_cell.v vs ref/lif_ref.py.
// Vectors from ref/lif_*.hex (regenerate with:  python3 ../ref/lif_ref.py).
// Verdict: "ALL PASS" / "FAIL (n mismatches)".  Uses !== (x-safe).
module tb_lif_cell;
    localparam integer MEMW = 20, INW = 12, ALPHA = 230, ASHIFT = 8, THRESH = 1024,
                       PW = 8, SURR = 5, T = 64;

    reg                    clk = 1'b0, rst_n = 1'b0, en = 1'b0, clr = 1'b0;
    reg  signed [INW-1:0]  in_cur = {INW{1'b0}};
    wire signed [MEMW-1:0] mem;
    wire                   spike;
    wire       [PW:0]      psi;

    lif_cell #(.MEMW(MEMW), .INW(INW), .ALPHA(ALPHA), .ASHIFT(ASHIFT), .THRESH(THRESH),
               .PW(PW), .SURR(SURR)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr), .in_cur(in_cur),
        .mem(mem), .spike(spike), .psi(psi));

    always #5 clk = ~clk;

    reg signed [INW-1:0]  in_v  [0:T-1];
    reg signed [MEMW-1:0] mem_e [0:T-1];
    reg        [0:0]      spk_e [0:T-1];
    reg        [PW:0]     psi_e [0:T-1];

    integer i, mism;
    initial begin
        $readmemh("../ref/lif_in.hex",      in_v);
        $readmemh("../ref/lif_mem_exp.hex", mem_e);
        $readmemh("../ref/lif_spk_exp.hex", spk_e);
        $readmemh("../ref/lif_psi_exp.hex", psi_e);

        // reset, then clear the membrane to 0
        @(negedge clk); @(negedge clk); rst_n = 1'b1;
        clr = 1'b1; @(negedge clk); clr = 1'b0;

        mism = 0;
        for (i = 0; i < T; i = i + 1) begin
            in_cur = in_v[i];
            en     = 1'b1;
            @(posedge clk);           // registers update: mem/spike now reflect step i
            #1;
            en = 1'b0;
            if (mem !== mem_e[i] || spike !== spk_e[i][0] || psi !== psi_e[i]) begin
                mism = mism + 1;
                $display("MISMATCH t=%0d in=%0d | mem %0d/%0d | spk %b/%b | psi %0d/%0d",
                         i, in_v[i], mem, mem_e[i], spike, spk_e[i][0], psi, psi_e[i]);
            end
            @(negedge clk);
        end

        if (mism == 0) $display("tb_lif_cell: ALL PASS (%0d steps, 0 mismatches)", T);
        else           $display("tb_lif_cell: FAIL (%0d mismatches)", mism);
        $finish;
    end
endmodule
`default_nettype wire
