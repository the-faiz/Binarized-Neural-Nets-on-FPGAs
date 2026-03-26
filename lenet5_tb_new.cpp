// ================================================================
// lenet5_tb.cpp
// Testbench — image comes from input_digit.h, no input argument
// ================================================================
#include <iostream>
#include <iomanip>
#include "lenet5_mnist.h"

int main()
{
    out_t result;
    int pass = 0, total = 0;

    std::cout << "================================================\n";
    std::cout << "  LeNet-5 Testbench\n";
    std::cout << "  Input digit (from input_digit.h): "
              << DIGIT_LABEL << "\n";
    std::cout << "================================================\n\n";

    // ── TEST 1: Run inference with hardcoded image ────────────────
    std::cout << "[TEST 1] Running inference on INPUT_IMAGE...\n";

    // Copy INPUT_IMAGE into a local mutable array
    // (lenet5_inference takes fm_t input[][][] not const)
    static fm_t img[IN_CH][IN_H][IN_W];
    for (int h = 0; h < IN_H; h++)
        for (int w = 0; w < IN_W; w++)
            img[0][h][w] = INPUT_IMAGE[0][h][w];

    // Print ASCII preview
    std::cout << "  Input preview (# = ON, . = OFF):\n";
    for (int h = 0; h < IN_H; h++) {
        std::cout << "  ";
        for (int w = 0; w < IN_W; w++)
            std::cout << ((float)img[0][h][w] > 0.3f ? '#' : '.');
        std::cout << "\n";
    }
    std::cout << "\n";

    lenet5_inference(img, result);

    std::cout << "  Output digit : " << (int)result << "\n";
    std::cout << "  4-bit binary : "
              << ((result>>3)&1) << ((result>>2)&1)
              << ((result>>1)&1) << (result&1) << "\n";
    bool valid   = ((int)result <= 9);
    bool correct = ((int)result == DIGIT_LABEL);
    std::cout << "  In range     : " << (valid   ? "PASS" : "FAIL") << "\n";
    std::cout << "  Correct class: " << (correct ? "YES" :
              "NO (expected with random weights)") << "\n\n";
    if (valid) pass++;
    total++;

    // ── TEST 2: Consistency — run twice, must get same result ─────
    std::cout << "[TEST 2] Consistency check\n";
    out_t r1, r2;
    lenet5_inference(img, r1);
    lenet5_inference(img, r2);
    bool consistent = (r1 == r2);
    std::cout << "  Run 1: " << (int)r1
              << "   Run 2: " << (int)r2 << "\n";
    std::cout << "  Status : " << (consistent ? "PASS" : "FAIL") << "\n\n";
    if (consistent) pass++;
    total++;

    // ── Summary ───────────────────────────────────────────────────
    std::cout << "================================================\n";
    std::cout << "  RESULTS : " << pass << " / " << total << " passed\n";
    std::cout << "  To change digit: edit input_digit.h\n";
    std::cout << "================================================\n";

    return 0;
}