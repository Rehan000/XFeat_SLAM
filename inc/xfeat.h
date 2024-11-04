#ifndef XFEAT_H
#define XFEAT_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "model_inference.h"
#include "interpolate_sparse_2d.h"
#include <torch/torch.h> 

// Struct to store keypoint data
struct KeypointData {
    torch::Tensor keypoints;    // N x 2 matrix for keypoint coordinates
    torch::Tensor scores;       // N x 1 vector for keypoint scores
    torch::Tensor descriptors;  // N x 64 matrix for feature descriptors
};

// Class definition for XFeat
class XFeat {
public:
    // Constructor to initialize the XFeat model
    XFeat(const std::string& model_path, int top_k = 4096, float detection_threshold = 0.05);

    // Function to detect and compute keypoints and descriptors
    std::vector<KeypointData> detectAndCompute(const cv::Mat& input_image);

    // Function to convert K1 logits into a heatmap
    torch::Tensor getKptsHeatmap(const torch::Tensor& K1, float softmax_temp = 1.0);

    // After generating K1h (heatmap), apply NMS to extract keypoints
    torch::Tensor NMS(const torch::Tensor& x, float threshold, int kernel_size);

    // Helper function to print tensor shape
    void printTensorShape(const torch::Tensor& tensor, const std::string& tensor_name);

    // Function to perform feature matching between two images and return additional keypoint data
    std::tuple<torch::Tensor, torch::Tensor, std::vector<KeypointData>, std::vector<KeypointData>>
    match_xfeat(const cv::Mat& input_image_1, const cv::Mat& input_image_2, int top_k = 1, float min_cossim = -1.0);

    // Function to perform mutual nearest neighbor (MNN) matching
    std::pair<torch::Tensor, torch::Tensor> match(const torch::Tensor& feats1, const torch::Tensor& feats2, float min_cossim = 0.82);


private:
    ModelInference model_inference;
    InterpolateSparse2d interpolator;
    int top_k;
    float detection_threshold;
    torch::Device device;
};

#endif  // XFEAT_H
