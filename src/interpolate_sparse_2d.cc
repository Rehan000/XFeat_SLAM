#include "interpolate_sparse_2d.h"
#include <torch/torch.h>
#include <cmath>
#include <iostream>

InterpolateSparse2d::InterpolateSparse2d(const std::string& mode, bool align_corners, torch::Device device)
    : mode_(mode), align_corners_(align_corners), device_(device) {}

torch::Tensor InterpolateSparse2d::normgrid(const torch::Tensor& pos, int H, int W) {
    torch::Tensor normalized = 2.0 * (pos / torch::tensor({W - 1, H - 1}, torch::TensorOptions().device(device_).dtype(pos.dtype()))) - 1.0;
    return normalized.to(device_);
}

// Forward function
torch::Tensor InterpolateSparse2d::forward(const torch::Tensor& x, const torch::Tensor& pos, int H, int W) {
    torch::Tensor grid = normgrid(pos, H, W).unsqueeze(1);
    grid = grid.to(x.dtype());

    torch::Tensor sampled = torch::nn::functional::grid_sample(
        x.to(device_),
        grid.to(device_),
        torch::nn::functional::GridSampleFuncOptions().mode(getMode()).align_corners(align_corners_)
    );

    return sampled.permute({0, 2, 3, 1}).squeeze(2);
}

torch::nn::functional::GridSampleFuncOptions::mode_t InterpolateSparse2d::getMode() const {
    if (mode_ == "nearest") {
        return torch::kNearest;
    } else if (mode_ == "bilinear") {
        return torch::kBilinear;
    } else if (mode_ == "bicubic") {
        return torch::kBilinear;
    } else {
        std::cerr << "Unsupported interpolation mode: " << mode_ << ". Defaulting to bilinear." << std::endl;
        return torch::kBilinear;
    }
}