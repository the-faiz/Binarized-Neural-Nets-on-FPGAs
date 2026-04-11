`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.04.2026 19:29:03
// Design Name: 
// Module Name: line_buffer
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

module line_buffer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        pixel_valid,     // Strobe: shift when high
    input  wire [7:0]  pixel_in,        // Current pixel (row 0 head)
    output wire [39:0] tap_out          // 5 taps x 8b, row0..row4
);

    // -------------------------------------------------------------------------
    // 4 delay lines, each 32 pixels deep (SRL32 inferred per byte)
    // dl[0] = 1-row delay, dl[1] = 2-row, ..., dl[3] = 4-row
    // -------------------------------------------------------------------------
    reg [7:0] dl0 [0:31];
    reg [7:0] dl1 [0:31];
    reg [7:0] dl2 [0:31];
    reg [7:0] dl3 [0:31];

    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 32; i = i + 1) begin
                dl0[i] <= 8'h00;
                dl1[i] <= 8'h00;
                dl2[i] <= 8'h00;
                dl3[i] <= 8'h00;
            end
        end else if (pixel_valid) begin
            // Shift each delay line: new sample enters [0], old [31] falls out
            // dl0 receives pixel_in
            dl0[0] <= pixel_in;
            // dl1 receives dl0[31] (1-row delayed output)
            dl1[0] <= dl0[31];
            // dl2 receives dl1[31] (2-row delayed output)
            dl2[0] <= dl1[31];
            // dl3 receives dl2[31] (3-row delayed output)
            dl3[0] <= dl2[31];

            // Shift the rest of each line
            for (i = 1; i < 32; i = i + 1) begin
                dl0[i] <= dl0[i-1];
                dl1[i] <= dl1[i-1];
                dl2[i] <= dl2[i-1];
                dl3[i] <= dl3[i-1];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Tap outputs: row0=current pixel, row1..row4 = delayed rows
    // tap_out[7:0]   = row 0 (newest)
    // tap_out[15:8]  = row 1 (1 row old)
    // tap_out[23:16] = row 2 (2 rows old)
    // tap_out[31:24] = row 3 (3 rows old)
    // tap_out[39:32] = row 4 (4 rows old)
    // -------------------------------------------------------------------------
    assign tap_out[ 7: 0] = pixel_in;     // Row 0: current
    assign tap_out[15: 8] = dl0[31];      // Row 1: 1-row delay output
    assign tap_out[23:16] = dl1[31];      // Row 2: 2-row delay output
    assign tap_out[31:24] = dl2[31];      // Row 3: 3-row delay output
    assign tap_out[39:32] = dl3[31];      // Row 4: 4-row delay output

endmodule