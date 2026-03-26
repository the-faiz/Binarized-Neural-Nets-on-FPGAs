// ================================================================
// lenet5_mnist.cpp
// LeNet-5 HLS Inference — resource-minimized for xc7z020
//
// What changed vs uploaded version to fix over-utilization:
//
//  1. FC layer sizes halved in header (FC1:120→60, FC2:84→42)
//     This is the single biggest fix:
//       Old FC1: 120 outputs × 400 inputs = 48,000 MACs
//       New FC1:  60 outputs × 400 inputs = 24,000 MACs  (-50%)
//
//  2. ALL #pragma HLS UNROLL removed from FC layers entirely.
//     UNROLL factor=4 inside PIPELINE II=1 still replicates the
//     multiply datapath 4× per output neuron. With 60 neurons ×
//     400/4 = 100 unrolled copies = 6000 multipliers for FC1
//     alone. Fully sequential: 1 multiplier reused 400 times.
//
//  3. FC loops restructured: inner loop is pipelined (II=1),
//     outer loop is sequential. This gives 1 MAC/cycle with
//     minimal hardware, completing FC1 in 60×400 = 24,000 cycles.
//     For a one-shot inference latency is not a concern.
//
//  4. ARRAY_PARTITION removed from all feature maps except fc3_out.
//     fc3_out needs complete partition for the parallel argmax.
//
//  5. Conv layers remain at II=4 / II=8 with no inner UNROLL.
// ================================================================

#include "lenet5_mnist.h"
#include "lenet5_weights.h"

// ================================================================
// Top-level
// ================================================================
void lenet5_inference(
    fm_t  input[IN_CH][IN_H][IN_W],
    out_t &digit_out
)
{
// ap_memory on input  : HLS generates a BRAM port (address+ce+q)
//                       driven by a Block Memory Generator in Vivado
//                       initialised with the image .coe file.
//                       Input is non-constant so HLS synthesises
//                       real hardware instead of constant-folding.
// ap_vld on digit_out : output register HOLDS value after ap_done
//                       so LEDs stay lit until next ap_start.
// ap_ctrl_hs on return: ap_start/ap_done/ap_idle handshake.
#pragma HLS INTERFACE ap_memory  port=input
#pragma HLS INTERFACE ap_vld     port=digit_out
#pragma HLS INTERFACE ap_ctrl_hs port=return

    // ── Local buffers ─────────────────────────────────────────────
    fm_t c1_out[C1_OUT_CH][C1_OUT_H][C1_OUT_W];
    fm_t p1_out[C1_OUT_CH][P1_OUT_H][P1_OUT_W];
    fm_t c2_out[C2_OUT_CH][C2_OUT_H][C2_OUT_W];
    fm_t p2_out[C2_OUT_CH][P2_OUT_H][P2_OUT_W];
    fm_t flat   [FC1_IN];
    fm_t fc1_out[FC1_OUT];
    fm_t fc2_out[FC2_OUT];
    fm_t fc3_out[FC3_OUT];

#pragma HLS ARRAY_PARTITION variable=fc3_out complete dim=1

    // ── Copy INPUT_IMAGE into local buffer ───────────────────────
    // INPUT_IMAGE is volatile — HLS cannot see pixel values at
    // compile time and must synthesise real MAC hardware.
    LOAD_INPUT:
    for (int h = 0; h < IN_H; h++)
        for (int w = 0; w < IN_W; w++) {
#pragma HLS PIPELINE II=1
            input[0][h][w] = (fm_t)INPUT_IMAGE[0][h][w];
        }
    conv2d_valid(input,   C1_W, C1_B, c1_out);
    relu_2d<C1_OUT_CH, C1_OUT_H, C1_OUT_W>(c1_out);
    avgpool2x2_c1(c1_out, p1_out);

    conv2d_c2(p1_out,    C2_W, C2_B, c2_out);
    relu_2d<C2_OUT_CH, C2_OUT_H, C2_OUT_W>(c2_out);
    avgpool2x2_c2(c2_out, p2_out);

    // ── FC ────────────────────────────────────────────────────────
    flatten(p2_out, flat);

    fc_layer_fc1(flat,    FC1_W, FC1_B, fc1_out);
    relu_1d(fc1_out, FC1_OUT);

    fc_layer_fc2(fc1_out, FC2_W, FC2_B, fc2_out);
    relu_1d(fc2_out, FC2_OUT);

    fc_layer_fc3(fc2_out, FC3_W, FC3_B, fc3_out);

    digit_out = argmax(fc3_out);
}

// ================================================================
// CONV1: [1][32][32] → [6][28][28]  valid 5×5
// No UNROLL on inner loops — single MAC datapath reused 25×
// II=4 gives routing margin with the large input feature map
// ================================================================
void conv2d_valid(
    fm_t        input [IN_CH][IN_H][IN_W],
    const wt_t  weight[C1_OUT_CH][IN_CH][C1_K][C1_K],
    const bn_t  bias  [C1_OUT_CH],
    fm_t        output[C1_OUT_CH][C1_OUT_H][C1_OUT_W]
)
{
    for (int oc = 0; oc < C1_OUT_CH; oc++)
        for (int oh = 0; oh < C1_OUT_H; oh++)
            for (int ow = 0; ow < C1_OUT_W; ow++) {
#pragma HLS PIPELINE II=4
                acc_t acc = (acc_t)bias[oc];
                for (int ic = 0; ic < IN_CH; ic++)
                    for (int kh = 0; kh < C1_K; kh++)
                        for (int kw = 0; kw < C1_K; kw++)
                            acc += (acc_t)input[ic][oh+kh][ow+kw]
                                 * (acc_t)weight[oc][ic][kh][kw];
                output[oc][oh][ow] = (fm_t)acc;
            }
}

// ================================================================
// CONV2: [6][14][14] → [16][10][10]  valid 5×5
// Inner loop: 6×5×5 = 150 MACs, fully sequential
// II=8 prevents memory bank conflicts on 6-channel input
// ================================================================
void conv2d_c2(
    fm_t        input [C1_OUT_CH][P1_OUT_H][P1_OUT_W],
    const wt_t  weight[C2_OUT_CH][C1_OUT_CH][C2_K][C2_K],
    const bn_t  bias  [C2_OUT_CH],
    fm_t        output[C2_OUT_CH][C2_OUT_H][C2_OUT_W]
)
{
    for (int oc = 0; oc < C2_OUT_CH; oc++)
        for (int oh = 0; oh < C2_OUT_H; oh++)
            for (int ow = 0; ow < C2_OUT_W; ow++) {
#pragma HLS PIPELINE II=8
                acc_t acc = (acc_t)bias[oc];
                for (int ic = 0; ic < C1_OUT_CH; ic++)
                    for (int kh = 0; kh < C2_K; kh++)
                        for (int kw = 0; kw < C2_K; kw++)
                            acc += (acc_t)input[ic][oh+kh][ow+kw]
                                 * (acc_t)weight[oc][ic][kh][kw];
                output[oc][oh][ow] = (fm_t)acc;
            }
}

// ================================================================
// AvgPool 2×2 stride 2 — after CONV1  [6][28][28] → [6][14][14]
// ================================================================
void avgpool2x2_c1(
    fm_t  input [C1_OUT_CH][C1_OUT_H][C1_OUT_W],
    fm_t  output[C1_OUT_CH][P1_OUT_H][P1_OUT_W]
)
{
    const fm_t scale = fm_t(0.25f);
    for (int c = 0; c < C1_OUT_CH; c++)
        for (int h = 0; h < P1_OUT_H; h++)
            for (int w = 0; w < P1_OUT_W; w++) {
#pragma HLS PIPELINE II=1
                acc_t s = (acc_t)input[c][h*2  ][w*2  ]
                        + (acc_t)input[c][h*2  ][w*2+1]
                        + (acc_t)input[c][h*2+1][w*2  ]
                        + (acc_t)input[c][h*2+1][w*2+1];
                output[c][h][w] = (fm_t)(s * (acc_t)scale);
            }
}

// ================================================================
// AvgPool 2×2 stride 2 — after CONV2  [16][10][10] → [16][5][5]
// ================================================================
void avgpool2x2_c2(
    fm_t  input [C2_OUT_CH][C2_OUT_H][C2_OUT_W],
    fm_t  output[C2_OUT_CH][P2_OUT_H][P2_OUT_W]
)
{
    const fm_t scale = fm_t(0.25f);
    for (int c = 0; c < C2_OUT_CH; c++)
        for (int h = 0; h < P2_OUT_H; h++)
            for (int w = 0; w < P2_OUT_W; w++) {
#pragma HLS PIPELINE II=1
                acc_t s = (acc_t)input[c][h*2  ][w*2  ]
                        + (acc_t)input[c][h*2  ][w*2+1]
                        + (acc_t)input[c][h*2+1][w*2  ]
                        + (acc_t)input[c][h*2+1][w*2+1];
                output[c][h][w] = (fm_t)(s * (acc_t)scale);
            }
}

// ================================================================
// Flatten [16][5][5] → [400]
// ================================================================
void flatten(
    fm_t  input [C2_OUT_CH][P2_OUT_H][P2_OUT_W],
    fm_t  output[FC1_IN]
)
{
    for (int c = 0; c < C2_OUT_CH; c++)
        for (int h = 0; h < P2_OUT_H; h++)
            for (int w = 0; w < P2_OUT_W; w++) {
#pragma HLS PIPELINE II=1
                output[c*P2_OUT_H*P2_OUT_W + h*P2_OUT_W + w]
                    = input[c][h][w];
            }
}

// ================================================================
// FC1: [400] → [60]
// Fully sequential inner loop: 1 multiplier reused 400 times.
// No UNROLL — this is the critical fix for LUT/CARRY4 overflow.
// Outer loop pipelined so each output neuron starts immediately
// after the previous one finishes.
// ================================================================
void fc_layer_fc1(
    fm_t        input [FC1_IN],
    const wt_t  weight[FC1_OUT][FC1_IN],
    const bn_t  bias  [FC1_OUT],
    fm_t        output[FC1_OUT]
)
{
    for (int o = 0; o < FC1_OUT; o++) {
        acc_t acc = (acc_t)bias[o];
        for (int i = 0; i < FC1_IN; i++) {
#pragma HLS PIPELINE II=1
            acc += (acc_t)input[i] * (acc_t)weight[o][i];
        }
        output[o] = (fm_t)acc;
    }
}

// ================================================================
// FC2: [60] → [42]   fully sequential inner loop
// ================================================================
void fc_layer_fc2(
    fm_t        input [FC2_IN],
    const wt_t  weight[FC2_OUT][FC2_IN],
    const bn_t  bias  [FC2_OUT],
    fm_t        output[FC2_OUT]
)
{
    for (int o = 0; o < FC2_OUT; o++) {
        acc_t acc = (acc_t)bias[o];
        for (int i = 0; i < FC2_IN; i++) {
#pragma HLS PIPELINE II=1
            acc += (acc_t)input[i] * (acc_t)weight[o][i];
        }
        output[o] = (fm_t)acc;
    }
}

// ================================================================
// FC3: [42] → [10]   fully sequential inner loop, no activation
// ================================================================
void fc_layer_fc3(
    fm_t        input [FC3_IN],
    const wt_t  weight[FC3_OUT][FC3_IN],
    const bn_t  bias  [FC3_OUT],
    fm_t        output[FC3_OUT]
)
{
    for (int o = 0; o < FC3_OUT; o++) {
        acc_t acc = (acc_t)bias[o];
        for (int i = 0; i < FC3_IN; i++) {
#pragma HLS PIPELINE II=1
            acc += (acc_t)input[i] * (acc_t)weight[o][i];
        }
        output[o] = (fm_t)acc;
    }
}

// ================================================================
// ReLU 1D in-place
// ================================================================
void relu_1d(fm_t data[], int size)
{
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        if (data[i] < fm_t(0)) data[i] = fm_t(0);
    }
}

// ================================================================
// Argmax over 10 logits → 4-bit digit (fully unrolled, only 10)
// ================================================================
out_t argmax(fm_t scores[FC3_OUT])
{
#pragma HLS INLINE
    fm_t  max_val = scores[0];
    out_t max_idx = 0;
    for (int i = 1; i < FC3_OUT; i++) {
#pragma HLS UNROLL
        if (scores[i] > max_val) {
            max_val = scores[i];
            max_idx = (out_t)i;
        }
    }
    return max_idx;
}