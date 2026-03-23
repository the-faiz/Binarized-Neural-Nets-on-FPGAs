// ============================================================
// conv_block.cpp
// HLS Convolution Block for CNN
// Compatible with: Xilinx Vivado HLS / Vitis HLS
// ============================================================

#include "conv_block.h"

// ============================================================
// Top-level Convolution Block
// Performs: Conv2D -> BatchNorm -> ReLU
// ============================================================
void conv_block(
    fm_t    input[IN_CH][IN_H][IN_W],
    wt_t    weights[OUT_CH][IN_CH][K][K],
    wt_t    bias[OUT_CH],
    bn_t    bn_gamma[OUT_CH],
    bn_t    bn_beta[OUT_CH],
    bn_t    bn_mean[OUT_CH],
    bn_t    bn_var[OUT_CH],
    fm_t    output[OUT_CH][OUT_H][OUT_W]
)
{
#pragma HLS INTERFACE s_axilite port=return          bundle=CTRL
#pragma HLS INTERFACE m_axi     port=input           offset=slave bundle=MEM0 depth=IN_CH*IN_H*IN_W
#pragma HLS INTERFACE m_axi     port=weights         offset=slave bundle=MEM1 depth=OUT_CH*IN_CH*K*K
#pragma HLS INTERFACE m_axi     port=bias            offset=slave bundle=MEM2 depth=OUT_CH
#pragma HLS INTERFACE m_axi     port=bn_gamma        offset=slave bundle=MEM3 depth=OUT_CH
#pragma HLS INTERFACE m_axi     port=bn_beta         offset=slave bundle=MEM3 depth=OUT_CH
#pragma HLS INTERFACE m_axi     port=bn_mean         offset=slave bundle=MEM3 depth=OUT_CH
#pragma HLS INTERFACE m_axi     port=bn_var          offset=slave bundle=MEM3 depth=OUT_CH
#pragma HLS INTERFACE m_axi     port=output          offset=slave bundle=MEM4 depth=OUT_CH*OUT_H*OUT_W

    // Local buffers for pipeline efficiency
    fm_t local_input[IN_CH][IN_H][IN_W];
    wt_t local_weights[OUT_CH][IN_CH][K][K];
    wt_t local_bias[OUT_CH];
    bn_t local_gamma[OUT_CH], local_beta[OUT_CH];
    bn_t local_mean[OUT_CH],  local_var[OUT_CH];
    fm_t conv_out[OUT_CH][OUT_H][OUT_W];
    fm_t local_output[OUT_CH][OUT_H][OUT_W];

#pragma HLS ARRAY_PARTITION variable=local_weights  cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=local_input    cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=conv_out       cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=local_gamma    complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_beta     complete dim=1

    // ----- Load inputs from external memory -----
    LOAD_INPUT:
    for (int c = 0; c < IN_CH; c++)
        for (int h = 0; h < IN_H; h++)
#pragma HLS PIPELINE II=1
            for (int w = 0; w < IN_W; w++)
                local_input[c][h][w] = input[c][h][w];

    LOAD_WEIGHTS:
    for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_CH; ic++)
            for (int kh = 0; kh < K; kh++)
#pragma HLS PIPELINE II=1
                for (int kw = 0; kw < K; kw++)
                    local_weights[oc][ic][kh][kw] = weights[oc][ic][kh][kw];

    LOAD_BN_PARAMS:
    for (int oc = 0; oc < OUT_CH; oc++) {
#pragma HLS PIPELINE II=1
        local_bias[oc]  = bias[oc];
        local_gamma[oc] = bn_gamma[oc];
        local_beta[oc]  = bn_beta[oc];
        local_mean[oc]  = bn_mean[oc];
        local_var[oc]   = bn_var[oc];
    }

    // ----- Step 1: Convolution with zero-padding -----
    conv2d(local_input, local_weights, local_bias, conv_out);

    // ----- Step 2: Batch Normalization -----
    batch_norm(conv_out, local_gamma, local_beta, local_mean, local_var, local_output);

    // ----- Step 3: ReLU Activation -----
    relu(local_output);

    // ----- Store output to external memory -----
    STORE_OUTPUT:
    for (int oc = 0; oc < OUT_CH; oc++)
        for (int h = 0; h < OUT_H; h++)
#pragma HLS PIPELINE II=1
            for (int w = 0; w < OUT_W; w++)
                output[oc][h][w] = local_output[oc][h][w];
}


// ============================================================
// Conv2D: 2D Convolution with same-padding (zero-pad)
// ============================================================
void conv2d(
    fm_t input[IN_CH][IN_H][IN_W],
    wt_t weights[OUT_CH][IN_CH][K][K],
    wt_t bias[OUT_CH],
    fm_t output[OUT_CH][OUT_H][OUT_W]
)
{


    CONV_OC:
    for (int oc = 0; oc < OUT_CH; oc++) {
        CONV_OH:
        for (int oh = 0; oh < OUT_H; oh++) {
            CONV_OW:
            for (int ow = 0; ow < OUT_W; ow++) {
#pragma HLS PIPELINE II=1

                acc_t acc = (acc_t)bias[oc];

                CONV_IC:
                for (int ic = 0; ic < IN_CH; ic++) {
                    CONV_KH:
                    for (int kh = 0; kh < K; kh++) {
                        CONV_KW:
                        for (int kw = 0; kw < K; kw++) {
#pragma HLS UNROLL

                            int ih = oh * STRIDE - PAD + kh;
                            int iw = ow * STRIDE - PAD + kw;

                            fm_t pixel = 0;
                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                                pixel = input[ic][ih][iw];

                            acc += (acc_t)pixel * (acc_t)weights[oc][ic][kh][kw];
                        }
                    }
                }

                // Quantize accumulator back to feature map precision
                output[oc][oh][ow] = (fm_t)acc;
            }
        }
    }
}


// ============================================================
// Batch Normalization (inference mode)
// y = gamma * (x - mean) / sqrt(var + eps) + beta
// ============================================================
void batch_norm(
    fm_t input[OUT_CH][OUT_H][OUT_W],
    bn_t gamma[OUT_CH],
    bn_t beta[OUT_CH],
    bn_t mean[OUT_CH],
    bn_t var[OUT_CH],
    fm_t output[OUT_CH][OUT_H][OUT_W]
)
{
    BN_C:
    for (int c = 0; c < OUT_CH; c++) {

        // Pre-compute scale and shift per channel
        bn_t inv_std = bn_t(1.0f) / hls::sqrt(var[c] + bn_t(BN_EPS));
        bn_t scale   = gamma[c] * inv_std;
        bn_t shift   = beta[c]  - scale * mean[c];

        BN_H:
        for (int h = 0; h < OUT_H; h++) {
            BN_W:
            for (int w = 0; w < OUT_W; w++) {
#pragma HLS PIPELINE II=1
                output[c][h][w] = (fm_t)(scale * (bn_t)input[c][h][w] + shift);
            }
        }
    }
}


// ============================================================
// ReLU Activation (in-place)
// ============================================================
void relu(fm_t feature_map[OUT_CH][OUT_H][OUT_W])
{
    RELU_C:
    for (int c = 0; c < OUT_CH; c++) {
        RELU_H:
        for (int h = 0; h < OUT_H; h++) {
            RELU_W:
            for (int w = 0; w < OUT_W; w++) {
#pragma HLS PIPELINE II=1
                if (feature_map[c][h][w] < fm_t(0))
                    feature_map[c][h][w] = fm_t(0);
            }
        }
    }
}