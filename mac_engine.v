`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2026 19:28:08
// Design Name: 
// Module Name: mac_engine
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

module mac_engine (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,       // Synchronous clear for accumulator
    input  wire        valid_in,    // Gate: accumulate only when high
    input  wire signed [7:0]  a,   // 8-bit signed input (activation / pixel)
    input  wire signed [7:0]  b,   // 8-bit signed weight
    output wire signed [23:0] acc_out  // 24-bit signed accumulator output
);

    // -------------------------------------------------------------------------
    // Stage 1: Registered multiply (maps to DSP48E1 MREG=1)
    // -------------------------------------------------------------------------
    reg signed [15:0] product_reg;

    always @(posedge clk) begin
        if (!rst_n)
            product_reg <= 16'sd0;
        else
            product_reg <= a * b;  // 8x8 signed multiply -> 16b
    end

    // -------------------------------------------------------------------------
    // Stage 2: Accumulator (maps to DSP48E1 PREG + OPMODE feedback)
    // -------------------------------------------------------------------------
    reg signed [23:0] accumulator;

    always @(posedge clk) begin
        if (!rst_n)
            accumulator <= 24'sd0;
        else if (clear)
            accumulator <= 24'sd0;
        else if (valid_in)
            accumulator <= accumulator + {{8{product_reg[15]}}, product_reg};
    end

    assign acc_out = accumulator;

endmodule