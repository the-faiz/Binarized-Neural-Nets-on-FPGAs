// ================================================================
// lenet5_mnist.h
// LeNet-5 HLS — MNIST digit recognition
// Resource-minimized for Zynq Z-7020 (53,200 LUT / 13,300 CARRY4)
//
// Key changes vs previous version:
//   - FC layer sizes halved: 120→60, 84→42
//   - acc_t kept at ap_fixed<24,10>
//   - All UNROLL pragmas removed from FC (done in .cpp)
//   - relu_2d template kept inline in header
// ================================================================
#ifndef LENET5_MNIST_H
#define LENET5_MNIST_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>

// ---- Input dimensions ----
#define IN_CH       1
#define IN_H        32
#define IN_W        32

// ---- Conv1 (5x5 valid): [1][32][32] -> [6][28][28] ----
#define C1_K        5
#define C1_OUT_CH   6
#define C1_OUT_H    28
#define C1_OUT_W    28

// ---- Pool1 (2x2): [6][28][28] -> [6][14][14] ----
#define P1_OUT_H    14
#define P1_OUT_W    14

// ---- Conv2 (5x5 valid): [6][14][14] -> [16][10][10] ----
#define C2_K        5
#define C2_OUT_CH   16
#define C2_OUT_H    10
#define C2_OUT_W    10

// ---- Pool2 (2x2): [16][10][10] -> [16][5][5] ----
#define P2_OUT_H    5
#define P2_OUT_W    5

// ---- FC layers — HALVED vs original LeNet-5 ----
// FC1: 400 -> 60   (was 120, saves 48000-24000=24000 weight MACs)
// FC2:  60 -> 42   (was 84)
// FC3:  42 -> 10   (output always 10)
#define FC1_IN      400
#define FC1_OUT     60
#define FC2_IN      60
#define FC2_OUT     42
#define FC3_IN      42
#define FC3_OUT     10

// ---- Fixed-point types ----
typedef ap_fixed<16, 6>   fm_t;   // feature map  Q6.10
typedef ap_fixed<16, 4>   wt_t;   // weights      Q4.12
typedef ap_fixed<24, 10>  acc_t;  // accumulator  Q10.14  (narrowed)
typedef ap_fixed<16, 6>   bn_t;   // bias         Q6.10
typedef ap_uint<4>         out_t; // 4-bit output 0-9

// ---- Top-level ----
void lenet5_inference(
    fm_t  input[IN_CH][IN_H][IN_W],
    out_t &digit_out
);

// ---- Sub-functions ----
void conv2d_valid(
    fm_t        input [IN_CH][IN_H][IN_W],
    const wt_t  weight[C1_OUT_CH][IN_CH][C1_K][C1_K],
    const bn_t  bias  [C1_OUT_CH],
    fm_t        output[C1_OUT_CH][C1_OUT_H][C1_OUT_W]
);

void conv2d_c2(
    fm_t        input [C1_OUT_CH][P1_OUT_H][P1_OUT_W],
    const wt_t  weight[C2_OUT_CH][C1_OUT_CH][C2_K][C2_K],
    const bn_t  bias  [C2_OUT_CH],
    fm_t        output[C2_OUT_CH][C2_OUT_H][C2_OUT_W]
);

void avgpool2x2_c1(
    fm_t  input [C1_OUT_CH][C1_OUT_H][C1_OUT_W],
    fm_t  output[C1_OUT_CH][P1_OUT_H][P1_OUT_W]
);

void avgpool2x2_c2(
    fm_t  input [C2_OUT_CH][C2_OUT_H][C2_OUT_W],
    fm_t  output[C2_OUT_CH][P2_OUT_H][P2_OUT_W]
);

void flatten(
    fm_t  input [C2_OUT_CH][P2_OUT_H][P2_OUT_W],
    fm_t  output[FC1_IN]
);

void fc_layer_fc1(
    fm_t        input [FC1_IN],
    const wt_t  weight[FC1_OUT][FC1_IN],
    const bn_t  bias  [FC1_OUT],
    fm_t        output[FC1_OUT]
);

void fc_layer_fc2(
    fm_t        input [FC2_IN],
    const wt_t  weight[FC2_OUT][FC2_IN],
    const bn_t  bias  [FC2_OUT],
    fm_t        output[FC2_OUT]
);

void fc_layer_fc3(
    fm_t        input [FC3_IN],
    const wt_t  weight[FC3_OUT][FC3_IN],
    const bn_t  bias  [FC3_OUT],
    fm_t        output[FC3_OUT]
);

void relu_1d(fm_t data[], int size);
out_t argmax(fm_t scores[FC3_OUT]);

// ---- Templated ReLU — defined here, used in .cpp ----
template<int CH, int H, int W>
void relu_2d(fm_t fm[CH][H][W])
{
    for (int c = 0; c < CH; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                if (fm[c][h][w] < fm_t(0)) fm[c][h][w] = fm_t(0);
            }
}

// ---- Input image (uses fm_t, so included after typedef) ----
#include "input_digit.h"

#endif // LENET5_MNIST_H