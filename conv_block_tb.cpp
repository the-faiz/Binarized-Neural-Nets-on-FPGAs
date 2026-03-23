// ============================================================
// conv_block_tb.cpp
// Testbench for HLS Convolution Block
// ============================================================

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "conv_block.h"

// Tolerance for fixed-point vs. float comparison
#define TOLERANCE 0.1f

// Simple software reference convolution (float)
void sw_conv2d_ref(
    float input[IN_CH][IN_H][IN_W],
    float weights[OUT_CH][IN_CH][K][K],
    float bias[OUT_CH],
    float output[OUT_CH][OUT_H][OUT_W]
)
{

    for (int oc = 0; oc < OUT_CH; oc++)
        for (int oh = 0; oh < OUT_H; oh++)
            for (int ow = 0; ow < OUT_W; ow++) {
                float acc = bias[oc];
                for (int ic = 0; ic < IN_CH; ic++)
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * STRIDE - PAD + kh;
                            int iw = ow * STRIDE - PAD + kw;
                            float px = (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                                       ? input[ic][ih][iw] : 0.0f;
                            acc += px * weights[oc][ic][kh][kw];
                        }
                output[oc][oh][ow] = acc;
            }
}

void sw_batchnorm_relu_ref(
    float inout[OUT_CH][OUT_H][OUT_W],
    float gamma[OUT_CH], float beta[OUT_CH],
    float mean[OUT_CH],  float var[OUT_CH]
)
{
    for (int c = 0; c < OUT_CH; c++) {
        float inv_std = 1.0f / std::sqrt(var[c] + BN_EPS);
        float scale   = gamma[c] * inv_std;
        float shift   = beta[c]  - scale * mean[c];
        for (int h = 0; h < OUT_H; h++)
            for (int w = 0; w < OUT_W; w++) {
                float val = scale * inout[c][h][w] + shift;
                inout[c][h][w] = val > 0.0f ? val : 0.0f; // ReLU
            }
    }
}

int main()
{
    std::cout << "===== HLS Conv Block Testbench =====" << std::endl;

    // ---- Allocate & randomise data ----
    static fm_t  hls_input  [IN_CH][IN_H][IN_W];
    static wt_t  hls_weights[OUT_CH][IN_CH][K][K];
    static wt_t  hls_bias   [OUT_CH];
    static bn_t  hls_gamma  [OUT_CH], hls_beta[OUT_CH];
    static bn_t  hls_mean   [OUT_CH], hls_var [OUT_CH];
    static fm_t  hls_output [OUT_CH][OUT_H][OUT_W];

    static float ref_input  [IN_CH][IN_H][IN_W];
    static float ref_weights[OUT_CH][IN_CH][K][K];
    static float ref_bias   [OUT_CH];
    static float ref_gamma  [OUT_CH], ref_beta[OUT_CH];
    static float ref_mean   [OUT_CH], ref_var [OUT_CH];
    static float ref_output [OUT_CH][OUT_H][OUT_W];

    srand(42);
    auto rnd = []() { return (float)(rand() % 200 - 100) / 100.0f; };

    for (int c = 0; c < IN_CH; c++)
        for (int h = 0; h < IN_H; h++)
            for (int w = 0; w < IN_W; w++) {
                float v = rnd();
                hls_input[c][h][w] = (fm_t)v;
                ref_input[c][h][w] = v;
            }

    for (int oc = 0; oc < OUT_CH; oc++) {
        ref_bias[oc] = rnd() * 0.1f;
        hls_bias[oc] = (wt_t)ref_bias[oc];
        ref_gamma[oc] = 0.9f + rnd()*0.1f;
        ref_beta [oc] = rnd()*0.1f;
        ref_mean [oc] = rnd()*0.5f;
        ref_var  [oc] = std::abs(rnd()) + 0.1f; // always positive
        hls_gamma[oc] = (bn_t)ref_gamma[oc];
        hls_beta [oc] = (bn_t)ref_beta [oc];
        hls_mean [oc] = (bn_t)ref_mean [oc];
        hls_var  [oc] = (bn_t)ref_var  [oc];

        for (int ic = 0; ic < IN_CH; ic++)
            for (int kh = 0; kh < K; kh++)
                for (int kw = 0; kw < K; kw++) {
                    float v = rnd() * 0.3f;
                    hls_weights[oc][ic][kh][kw] = (wt_t)v;
                    ref_weights[oc][ic][kh][kw] = v;
                }
    }

    // ---- Run HLS top-level ----
    conv_block(hls_input, hls_weights, hls_bias,
               hls_gamma, hls_beta, hls_mean, hls_var,
               hls_output);

    // ---- Run software reference ----
    sw_conv2d_ref(ref_input, ref_weights, ref_bias, ref_output);
    sw_batchnorm_relu_ref(ref_output, ref_gamma, ref_beta, ref_mean, ref_var);

    // ---- Compare results ----
    int errors = 0;
    float max_diff = 0.0f;

    for (int oc = 0; oc < OUT_CH; oc++)
        for (int h = 0; h < OUT_H; h++)
            for (int w = 0; w < OUT_W; w++) {
                float hls_val = (float)hls_output[oc][h][w];
                float ref_val = ref_output[oc][h][w];
                float diff    = std::abs(hls_val - ref_val);
                if (diff > max_diff) max_diff = diff;
                if (diff > TOLERANCE) {
                    errors++;
                    if (errors <= 5)
                        std::cout << "[MISMATCH] oc=" << oc
                                  << " h=" << h << " w=" << w
                                  << "  HLS=" << hls_val
                                  << "  REF=" << ref_val
                                  << "  diff=" << diff << std::endl;
                }
            }

    std::cout << "Max absolute error : " << max_diff << std::endl;
    std::cout << "Total mismatches   : " << errors
              << " / " << OUT_CH*OUT_H*OUT_W << std::endl;

    if (errors == 0)
        std::cout << "PASS: All outputs within tolerance." << std::endl;
    else
        std::cout << "FAIL: " << errors << " outputs exceeded tolerance." << std::endl;

    return errors;
}