`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2026 19:31:41
// Design Name: 
// Module Name: feature_bram
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module feature_bram #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10   // 2^10 = 1024 entries (>784, headroom for pool)
) (
    input  wire                  clk,
    // Port A: Write
    input  wire                  we_a,
    input  wire [ADDR_WIDTH-1:0] addr_a,
    input  wire [DATA_WIDTH-1:0] din_a,
    // Port B: Read
    input  wire [ADDR_WIDTH-1:0] addr_b,
    output reg  [DATA_WIDTH-1:0] dout_b
);

    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Port A: synchronous write
    always @(posedge clk) begin
        if (we_a)
            mem[addr_a] <= din_a;
    end

    // Port B: synchronous read
    always @(posedge clk) begin
        dout_b <= mem[addr_b];
    end

endmodule
