`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: weight_bram
// Description: Simple Dual-Port BRAM with File Initialization support.
//              Infers RAMB36E1 primitives on Xilinx 7-Series (Zedboard).
//////////////////////////////////////////////////////////////////////////////////

module weight_bram #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 15,  // 2^15 = 32768 entries of 8 bits
    parameter INIT_FILE  = "random_weights.mem" // Default initialization file
) (
    input  wire                  clk,
    // Write port (for dynamic updates if needed)
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [DATA_WIDTH-1:0] wr_data,
    // Read port
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output reg  [DATA_WIDTH-1:0] rd_data
);

    // -------------------------------------------------------------------------
    // Memory array declaration
    // -------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // -------------------------------------------------------------------------
    // BRAM Initialization for Synthesis and Simulation
    // -------------------------------------------------------------------------
    initial begin
        if (INIT_FILE != "") begin
            $display("Loading BRAM from %s", INIT_FILE);
            $readmemh(INIT_FILE, mem);
        end
    end

    // -------------------------------------------------------------------------
    // Synchronous Write Logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (we)
            mem[wr_addr] <= wr_data;
    end

    // -------------------------------------------------------------------------
    // Synchronous Read Logic (1-cycle latency)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        rd_data <= mem[rd_addr];
    end

endmodule