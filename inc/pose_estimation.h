// pose_estimation.h
#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/torch.h> 

class PoseEstimation {
public:
    // Constructor
    PoseEstimation(const cv::Mat& camera_intrinsics);

    // Estimate relative pose between two frames
    bool estimatePose(const torch::Tensor& pts_1, 
                      const torch::Tensor& pts_2,
                      cv::Mat& R, cv::Mat& t);

    // Convert Torch tensors to vector Point2f
    std::vector<cv::Point2f> convertToPoints(const torch::Tensor& tensor);

private:
    cv::Mat camera_intrinsics_; // Camera intrinsic matrix
};

#endif // POSE_ESTIMATION_H