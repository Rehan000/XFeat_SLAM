#ifndef INTERPOLATE_SPARSE_2D_H
#define INTERPOLATE_SPARSE_2D_H

#include <torch/torch.h>
#include <string>

class InterpolateSparse2d {
public:
    // Constructor with interpolation mode, align_corners flag, and device (CPU or GPU)
    InterpolateSparse2d(const std::string& mode = "bilinear", bool align_corners = false, torch::Device device = torch::kCPU);

    // Forward function to perform interpolation
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& pos, int H, int W);

private:
    std::string mode_;          // Interpolation mode: bilinear, nearest, or bicubic
    bool align_corners_;        // Whether to align corners for interpolation
    torch::Device device_;      // Device (GPU/CPU) to run the operations on

    // Helper function to normalize the 2D coordinates to the [-1, 1] range
    torch::Tensor normgrid(const torch::Tensor& pos, int H, int W);

    // Helper function to map mode string to libtorch's enum type
    torch::nn::functional::GridSampleFuncOptions::mode_t getMode() const;

};

#endif // INTERPOLATE_SPARSE_2D_H
