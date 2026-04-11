`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2026 19:32:38
// Design Name: 
// Module Name: relu_sat
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

module relu_sat #(
    parameter SHIFT = 6  // This must match the top module
)(
    input  wire signed [23:0] acc_in,
    output wire [7:0]         act_out
);
    // Use the triple-chevron (>>>) for signed arithmetic shift
    wire signed [23:0] scaled_acc = acc_in >>> SHIFT;

    assign act_out = (scaled_acc < 0)    ? 8'd0 : 
                     (scaled_acc > 127)  ? 8'd127 : 
                     scaled_acc[7:0];
endmodule