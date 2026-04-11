`timescale 1ns / 1ps
// =============================================================================
// cnn_lenet_zedboard.v  -  TOP-LEVEL MODULE (FINAL STABLE PIPELINE)
// =============================================================================

module cnn_lenet_zedboard (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  pixel_in,            
    input  wire        pixel_valid_in,      
    output wire [3:0]  classification_result,
    output wire        result_valid
);

    // =========================================================================
    // PARAMETERS
    // =========================================================================
    parameter IMG_W        = 32;
    parameter IMG_H        = 32;
    parameter FILT_SIZE    = 5;
    parameter N_FILTERS    = 4;
    parameter CONV_OUT_W   = 28;
    parameter CONV_OUT_H   = 28;
    parameter POOL_OUT_W   = 14;
    parameter POOL_OUT_H   = 14;
    parameter N_FC1        = 32;
    parameter N_FC2        = 10;
    
    // MATH TUNING: Shift 6 prevents saturation of large 8-bit image inputs
    parameter RELU_SHIFT   = 6;     

    parameter CONV_W_BASE  = 15'd0;
    parameter CONV_B_BASE  = 15'd100;
    parameter FC1_W_BASE   = 15'd104;
    parameter FC1_B_BASE   = 15'd25192;
    parameter FC2_W_BASE   = 15'd25224;
    parameter FC2_B_BASE   = 15'd25544;

    // FSM ENCODING
    parameter [3:0]
        S_IDLE       = 4'd0,  S_LOAD_IMG   = 4'd1,  S_CONV_PREP  = 4'd2,
        S_CONV_LOAD  = 4'd3,  S_CONV_WAIT  = 4'd4,  S_CONV_STORE = 4'd5,
        S_POOL_CALC  = 4'd6,  S_POOL_STORE = 4'd7,  S_FC1_PREP   = 4'd8,
        S_FC1_LOAD   = 4'd9,  S_FC1_STORE  = 4'd10, S_FC2_PREP   = 4'd11,
        S_FC2_LOAD   = 4'd12, S_FC2_STORE  = 4'd13, S_DONE       = 4'd14;

    reg [3:0] state;

    // =========================================================================
    // REGISTERS & MEMORIES
    // =========================================================================
    reg [9:0]  pixel_count;   
    reg [7:0]  img_buf [0:1023]; 

    reg [4:0]  conv_out_row, conv_out_col;
    reg        conv_store_phase;
    reg [1:0]  conv_filt;
    
    // Decoupled Convolution Pipeline Counters
    reg [2:0]  req_kr, req_kc, cons_kr, cons_kc;
    reg [1:0]  req_filt, cons_filt;
    reg [7:0]  conv_img_pipe [0:1];
    
    reg [14:0] wb_rd_addr;
    wire [7:0] wb_rd_data;

    weight_bram #(.DATA_WIDTH(8), .ADDR_WIDTH(15), .INIT_FILE("random_weights.mem")) u_weight_bram (
        .clk(clk), .we(1'b0), .wr_addr(15'd0), .wr_data(8'd0), .rd_addr(wb_rd_addr), .rd_data(wb_rd_data)
    );

    wire signed [23:0] mac_acc [0:3];
    reg  [7:0]         mac_a [0:3], mac_b [0:3];
    reg                mac_clr [0:3], mac_vld [0:3];

    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : mac_array
            mac_engine u_mac (
                .clk(clk), .rst_n(rst_n), .clear(mac_clr[gi]), .valid_in(mac_vld[gi]), 
                .a(mac_a[gi]), .b(mac_b[gi]), .acc_out(mac_acc[gi])
            );
        end
    endgenerate

    reg signed [23:0] relu_in  [0:3];
    wire [7:0]        relu_out [0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : relu_array
            relu_sat #(.SHIFT(RELU_SHIFT)) u_relu (
                .acc_in(relu_in[gi]), .act_out(relu_out[gi])
            );
        end
    endgenerate

    reg        feat_we [0:3];
    reg [9:0]  feat_waddr [0:3], feat_raddr [0:3];
    reg [7:0]  feat_wdata [0:3];
    wire [7:0] feat_rdata [0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : feat_brams
            feature_bram #(.DATA_WIDTH(8), .ADDR_WIDTH(10)) u_feat (
                .clk(clk), .we_a(feat_we[gi]), .addr_a(feat_waddr[gi]), .din_a(feat_wdata[gi]), 
                .addr_b(feat_raddr[gi]), .dout_b(feat_rdata[gi])
            );
        end
    endgenerate

    reg        pool_we [0:3];
    reg [7:0]  pool_waddr [0:3], pool_wdata [0:3], pool_raddr [0:3];
    wire [7:0] pool_rdata [0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : pool_brams
            feature_bram #(.DATA_WIDTH(8), .ADDR_WIDTH(8)) u_pool (
                .clk(clk), .we_a(pool_we[gi]), .addr_a(pool_waddr[gi]), .din_a(pool_wdata[gi]), 
                .addr_b(pool_raddr[gi]), .dout_b(pool_rdata[gi])
            );
        end
    endgenerate

    reg [7:0] fc1_act [0:31];
    reg signed [23:0] fc2_logit [0:9];
    reg [3:0]         argmax_class;
    reg signed [23:0] argmax_val;

    reg [3:0]  pool_row, pool_col;
    reg [1:0]  pool_filt, pool_sub_r, pool_sub_c;
    reg [7:0]  pool_max;
    reg        pool_addr_issued;

    reg [4:0]  fc1_neuron;
    reg [9:0]  fc1_input;
    reg [3:0]  fc2_class;
    reg [5:0]  fc2_input;
    
    // Upgraded to 7 bits to handle the 100+ cycle decoupled Conv pipeline
    reg [6:0]  mac_pipe_cnt; 

    reg [3:0]  result_reg;
    reg        result_valid_reg;

    assign classification_result = result_reg;
    assign result_valid          = result_valid_reg;

    wire [9:0] conv_out_addr = conv_out_row * CONV_OUT_W + conv_out_col;
    wire [7:0] pool_out_addr = pool_row * POOL_OUT_W + pool_col;

    // =========================================================================
    // MAIN FSM
    // =========================================================================
    integer j;
    always @(posedge clk) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            pixel_count      <= 10'd0;
            conv_out_row     <= 5'd0;
            conv_out_col     <= 5'd0;
            conv_filt        <= 2'd0;
            conv_store_phase <= 1'b0;
            req_kr <= 0; req_kc <= 0; req_filt <= 0;
            cons_kr <= 0; cons_kc <= 0; cons_filt <= 0;
            pool_row         <= 4'd0;
            pool_col         <= 4'd0;
            pool_filt        <= 2'd0;
            pool_sub_r       <= 2'd0;
            pool_sub_c       <= 2'd0;
            pool_max         <= 8'd0;
            pool_addr_issued <= 1'b0;
            fc1_neuron       <= 5'd0;
            fc1_input        <= 10'd0;
            fc2_class        <= 4'd0;
            fc2_input        <= 6'd0;
            mac_pipe_cnt     <= 7'd0;
            argmax_class     <= 4'd0;
            argmax_val       <= 24'sd0;
            result_reg       <= 4'd0;
            result_valid_reg <= 1'b0;
            wb_rd_addr       <= 15'd0;
            conv_img_pipe[0] <= 8'd0;
            conv_img_pipe[1] <= 8'd0;

            for (j = 0; j < 4; j = j + 1) begin
                mac_a[j]      <= 8'd0;
                mac_b[j]      <= 8'd0;
                mac_clr[j]    <= 1'b0;
                mac_vld[j]    <= 1'b0;
                relu_in[j]    <= 24'sd0;
                feat_we[j]    <= 1'b0;
                feat_waddr[j] <= 10'd0;
                feat_wdata[j] <= 8'd0;
                feat_raddr[j] <= 10'd0;
                pool_we[j]    <= 1'b0;
                pool_waddr[j] <= 8'd0;
                pool_wdata[j] <= 8'd0;
                pool_raddr[j] <= 8'd0;
            end
            for (j = 0; j < 1024; j = j + 1) img_buf[j] <= 8'd0;

        end else begin
            for (j = 0; j < 4; j = j + 1) begin
                feat_we[j] <= 0; pool_we[j] <= 0; mac_clr[j] <= 0; mac_vld[j] <= 0;
            end

            case (state)
                S_IDLE: if (pixel_valid_in) begin img_buf[0] <= pixel_in; pixel_count <= 1; state <= S_LOAD_IMG; end
                
                S_LOAD_IMG: begin
                    if (pixel_valid_in) begin
                        img_buf[pixel_count] <= pixel_in;
                        if (pixel_count == 1023) state <= S_CONV_PREP;
                        else pixel_count <= pixel_count + 1;
                    end
                end

                S_CONV_PREP: begin
                    for (j = 0; j < 4; j = j + 1) mac_clr[j] <= 1;
                    req_kr <= 0; req_kc <= 0; req_filt <= 0;
                    cons_kr <= 0; cons_kc <= 0; cons_filt <= 0;
                    mac_pipe_cnt <= 0; 
                    conv_store_phase <= 0;
                    state <= S_CONV_LOAD;
                end

                S_CONV_LOAD: begin
                    // --- PHASE 1: REQUEST DATA (Cycles 0 to 99) ---
                    if (mac_pipe_cnt < 100) begin
                        wb_rd_addr <= CONV_W_BASE + req_filt*25 + req_kr*5 + req_kc;
                        conv_img_pipe[0] <= img_buf[(conv_out_row + req_kr) * 32 + (conv_out_col + req_kc)];

                        if (req_filt == 3) begin
                            req_filt <= 0;
                            if (req_kc == 4) begin
                                req_kc <= 0;
                                req_kr <= req_kr + 1;
                            end else req_kc <= req_kc + 1;
                        end else req_filt <= req_filt + 1;
                    end

                    // Shift pipeline to match 2-cycle BRAM latency
                    conv_img_pipe[1] <= conv_img_pipe[0];

                    // --- PHASE 2: CONSUME DATA (Cycles 2 to 101) ---
                    if (mac_pipe_cnt > 1 && mac_pipe_cnt <= 101) begin
                        mac_a[cons_filt] <= conv_img_pipe[1];
                        mac_b[cons_filt] <= wb_rd_data;
                        mac_vld[cons_filt] <= 1;

                        if (cons_filt == 3) begin
                            cons_filt <= 0;
                            if (cons_kc == 4) begin
                                cons_kc <= 0;
                                cons_kr <= cons_kr + 1;
                            end else cons_kc <= cons_kc + 1;
                        end else cons_filt <= cons_filt + 1;
                    end

                    mac_pipe_cnt <= mac_pipe_cnt + 1;
                    if (mac_pipe_cnt == 101) begin 
                        mac_pipe_cnt <= 0; 
                        state <= S_CONV_WAIT; 
                    end
                end

                S_CONV_WAIT: begin
                    mac_pipe_cnt <= mac_pipe_cnt + 1;
                    if (mac_pipe_cnt == 1) begin wb_rd_addr <= CONV_B_BASE; state <= S_CONV_STORE; conv_filt <= 0; conv_store_phase <= 0; end
                end

                S_CONV_STORE: begin
                    if (conv_store_phase == 0) begin
                        relu_in[conv_filt] <= mac_acc[conv_filt] + $signed(wb_rd_data);
                        conv_store_phase <= 1;
                        if (conv_filt < 3) wb_rd_addr <= CONV_B_BASE + conv_filt + 1;
                    end else begin
                        feat_we[conv_filt] <= 1; feat_waddr[conv_filt] <= conv_out_addr; feat_wdata[conv_filt] <= relu_out[conv_filt];
                        conv_store_phase <= 0;
                        if (conv_filt < 3) conv_filt <= conv_filt + 1;
                        else begin
                            conv_filt <= 0;
                            if (conv_out_col == 27) begin
                                conv_out_col <= 0;
                                if (conv_out_row == 27) begin 
                                    pool_row <= 0; pool_col <= 0; pool_filt <= 0; 
                                    pool_sub_r <= 0; pool_sub_c <= 0; pool_max <= 0; 
                                    pool_addr_issued <= 0; state <= S_POOL_CALC; 
                                end
                                else begin conv_out_row <= conv_out_row + 1; state <= S_CONV_PREP; end
                            end else begin conv_out_col <= conv_out_col + 1; state <= S_CONV_PREP; end
                        end
                    end
                end

                S_POOL_CALC: begin
                    if (!pool_addr_issued) begin
                        for (j = 0; j < 4; j = j + 1) feat_raddr[j] <= (pool_row*2 + pool_sub_r)*28 + (pool_col*2 + pool_sub_c);
                        pool_addr_issued <= 1;
                    end else begin
                        pool_addr_issued <= 0;
                        if (pool_sub_r == 0 && pool_sub_c == 0) pool_max <= feat_rdata[pool_filt];
                        else pool_max <= (feat_rdata[pool_filt] > pool_max) ? feat_rdata[pool_filt] : pool_max;

                        if (pool_sub_c == 1) begin
                            pool_sub_c <= 0;
                            if (pool_sub_r == 1) begin pool_sub_r <= 0; state <= S_POOL_STORE; end
                            else pool_sub_r <= pool_sub_r + 1;
                        end else pool_sub_c <= pool_sub_c + 1;
                    end
                end

                S_POOL_STORE: begin
                    pool_we[pool_filt] <= 1; pool_waddr[pool_filt] <= pool_out_addr; pool_wdata[pool_filt] <= pool_max;
                    if (pool_col == 13) begin
                        pool_col <= 0;
                        if (pool_row == 13) begin
                            if (pool_filt == 3) begin pool_filt <= 0; fc1_neuron <= 0; fc1_input <= 0; mac_clr[0] <= 1; state <= S_FC1_PREP; end
                            else begin pool_filt <= pool_filt + 1; pool_row <= 0; state <= S_POOL_CALC; end
                        end else begin pool_row <= pool_row + 1; state <= S_POOL_CALC; end
                    end else begin pool_col <= pool_col + 1; state <= S_POOL_CALC; end
                end

                S_FC1_PREP: begin fc1_neuron <= 0; fc1_input <= 0; mac_clr[0] <= 1; state <= S_FC1_LOAD; end

                S_FC1_LOAD: begin
                    // Address request (Cycle 0)
                    if (fc1_input < 784) begin
                        wb_rd_addr <= FC1_W_BASE + fc1_neuron * 784 + fc1_input;
                        if (fc1_input < 196) pool_raddr[0] <= fc1_input;
                        else if (fc1_input < 392) pool_raddr[1] <= fc1_input - 196;
                        else if (fc1_input < 588) pool_raddr[2] <= fc1_input - 392;
                        else                      pool_raddr[3] <= fc1_input - 588;
                    end
                    
                    fc1_input <= fc1_input + 1;

                    // Data Sample (Cycle 2, delay by 2 indices)
                    if (fc1_input > 1 && fc1_input <= 785) begin
                        mac_b[0] <= wb_rd_data; mac_vld[0] <= 1;
                        if      ((fc1_input-2) < 196) mac_a[0] <= pool_rdata[0];
                        else if ((fc1_input-2) < 392) mac_a[0] <= pool_rdata[1];
                        else if ((fc1_input-2) < 588) mac_a[0] <= pool_rdata[2];
                        else                          mac_a[0] <= pool_rdata[3];
                    end else begin
                        mac_vld[0] <= 0;
                    end

                    if (fc1_input == 785) begin mac_pipe_cnt <= 0; state <= S_FC1_STORE; end
                end

                S_FC1_STORE: begin
                    mac_pipe_cnt <= mac_pipe_cnt + 1;
                    
                    if (mac_pipe_cnt == 0) begin
                        wb_rd_addr <= FC1_B_BASE + fc1_neuron; 
                    end
                    else if (mac_pipe_cnt == 3) begin
                        relu_in[0] <= mac_acc[0] + $signed(wb_rd_data);
                    end
                    else if (mac_pipe_cnt == 4) begin 
                        fc1_act[fc1_neuron] <= relu_out[0];
                        if (fc1_neuron == 31) begin fc2_class <= 0; fc2_input <= 0; state <= S_FC2_PREP; end
                        else begin fc1_neuron <= fc1_neuron + 1; fc1_input <= 0; mac_clr[0] <= 1; state <= S_FC1_LOAD; end
                    end
                end

                S_FC2_PREP: begin mac_clr[0] <= 1; fc2_input <= 0; state <= S_FC2_LOAD; end

                S_FC2_LOAD: begin
                    // Address request (Cycle 0)
                    if (fc2_input < 32) begin 
                        wb_rd_addr <= FC2_W_BASE + fc2_class*32 + fc2_input[4:0]; 
                    end
                    fc2_input <= fc2_input + 1; 

                    // Data Sample (Cycle 2, delay by 2 indices)
                    if (fc2_input > 1 && fc2_input <= 33) begin 
                        mac_a[0] <= fc1_act[fc2_input-2]; 
                        mac_b[0] <= wb_rd_data; 
                        mac_vld[0] <= 1; 
                    end else begin
                        mac_vld[0] <= 0;
                    end

                    if (fc2_input == 33) begin mac_pipe_cnt <= 0; state <= S_FC2_STORE; end
                end

                S_FC2_STORE: begin
                    mac_pipe_cnt <= mac_pipe_cnt + 1;

                    if (mac_pipe_cnt == 0) begin
                        wb_rd_addr <= FC2_B_BASE + fc2_class;
                    end
                    else if (mac_pipe_cnt == 3) begin
                        fc2_logit[fc2_class] <= mac_acc[0] + $signed(wb_rd_data);
                    end
                    else if (mac_pipe_cnt == 4) begin
                        if (fc2_class == 0 || fc2_logit[fc2_class] > argmax_val) begin
                            argmax_class <= fc2_class; argmax_val <= fc2_logit[fc2_class];
                        end
                        if (fc2_class == 9) begin 
                            result_reg <= (fc2_logit[9] > argmax_val) ? 9 : argmax_class; 
                            result_valid_reg <= 1; state <= S_DONE; 
                        end else begin 
                            fc2_class <= fc2_class + 1; fc2_input <= 0; mac_clr[0] <= 1; state <= S_FC2_LOAD; 
                        end
                    end
                end

                S_DONE: if (pixel_valid_in) begin result_valid_reg <= 0; pixel_count <= 1; img_buf[0] <= pixel_in; state <= S_LOAD_IMG; end
            endcase
        end
    end
endmodule