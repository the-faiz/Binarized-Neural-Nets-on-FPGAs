// ============================================================
// conv_block.h
// Header for HLS Convolution Block
// ============================================================

#ifndef CONV_BLOCK_H
#define CONV_BLOCK_H

#include <ap_fixed.h>     // Xilinx arbitrary-precision fixed-point
#include <hls_math.h>     // HLS-safe math (hls::sqrt, etc.)

// ============================================================
// Network Hyper-Parameters  (edit these to change layer size)
// ============================================================

// Input feature map dimensions
#define IN_CH   3       // Input channels  (e.g. RGB)
#define IN_H    32      // Input height
#define IN_W    32      // Input width

// Convolution parameters
#define K       3       // Kernel size (3x3)
#define STRIDE  1       // Stride
#define PAD     (K/2)   // 'same' padding

// Output feature map dimensions  (same-padding, stride=1)
#define OUT_CH  16      // Number of filters
#define OUT_H   ((IN_H - K + 2*PAD) / STRIDE + 1)
#define OUT_W   ((IN_W - K + 2*PAD) / STRIDE + 1)

// Batch-norm epsilon
#define BN_EPS  1e-5f

// ============================================================
// Fixed-Point Data Types
// Adjust word-length (W) and integer-bits (I) for your design.
// ============================================================

// Feature-map type: Q8.8  (16-bit, 8 integer bits)
typedef ap_fixed<16, 8>  fm_t;

// Weight type: Q4.12 (16-bit, 4 integer bits)
typedef ap_fixed<16, 4>  wt_t;

// Accumulator type: wider to prevent overflow during MAC
typedef ap_fixed<32, 16> acc_t;

// Batch-norm parameter type (float-like precision)
typedef ap_fixed<24, 8>  bn_t;

// ============================================================
// Function Declarations
// ============================================================

// Top-level synthesisable function
void conv_block(
    fm_t  input   [IN_CH] [IN_H][IN_W],
    wt_t  weights [OUT_CH][IN_CH][K][K],
    wt_t  bias    [OUT_CH],
    bn_t  bn_gamma[OUT_CH],
    bn_t  bn_beta [OUT_CH],
    bn_t  bn_mean [OUT_CH],
    bn_t  bn_var  [OUT_CH],
    fm_t  output  [OUT_CH][OUT_H][OUT_W]
);

// Sub-functions (called internally, not synthesised separately)
void conv2d(
    fm_t  input  [IN_CH][IN_H][IN_W],
    wt_t  weights[OUT_CH][IN_CH][K][K],
    wt_t  bias   [OUT_CH],
    fm_t  output [OUT_CH][OUT_H][OUT_W]
);

void batch_norm(
    fm_t  input [OUT_CH][OUT_H][OUT_W],
    bn_t  gamma [OUT_CH],
    bn_t  beta  [OUT_CH],
    bn_t  mean  [OUT_CH],
    bn_t  var   [OUT_CH],
    fm_t  output[OUT_CH][OUT_H][OUT_W]
);

void relu(fm_t feature_map[OUT_CH][OUT_H][OUT_W]);

#endif // CONV_BLOCK_H