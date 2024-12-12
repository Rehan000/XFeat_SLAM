// pose_estimation.cc
#include "pose_estimation.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

// Constructor
PoseEstimation::PoseEstimation(const cv::Mat& camera_intrinsics) 
    : camera_intrinsics_(camera_intrinsics.clone()) {}

// Estimate relative pose between two frames
bool PoseEstimation::estimatePose(const torch::Tensor& pts_1, 
                                  const torch::Tensor& pts_2,
                                  cv::Mat& R, cv::Mat& t) {
    
    std::vector<cv::Point2f> points1 = convertToPoints(pts_1);
    std::vector<cv::Point2f> points2 = convertToPoints(pts_2);

    if (points1.size() < 8 || points2.size() < 8) {
        std::cerr << "Error: Not enough points for pose estimation." << std::endl;
        return false;
    }

    // Compute the essential matrix
    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, camera_intrinsics_, cv::RANSAC, 0.999, 1.0);

    if (essential_matrix.empty()) {
        std::cerr << "Error: Essential matrix computation failed." << std::endl;
        return false;
    }

    // Recover pose (rotation and translation)
    int inliers = cv::recoverPose(essential_matrix, points1, points2, camera_intrinsics_, R, t);

    if (inliers < 8) {
        std::cerr << "Error: Insufficient inliers for reliable pose estimation." << std::endl;
        return false;
    }

    // std::cout << "Pose estimation successful with " << inliers << " inliers." << std::endl;
    return true;
}

// Convert Torch tensors to vector Point2f
std::vector<cv::Point2f> PoseEstimation::convertToPoints(const torch::Tensor& tensor) {
    // Ensure tensor is on CPU and contiguous
    auto tensor_cpu = tensor.to(torch::kCPU).contiguous();

    // Access tensor data
    const float* data_ptr = tensor_cpu.data_ptr<float>();

    // Create cv::Point2f vector
    std::vector<cv::Point2f> points;
    points.reserve(tensor.size(0));
    for (int i = 0; i < tensor.size(0); ++i) {
        points.emplace_back(data_ptr[2 * i], data_ptr[2 * i + 1]);
    }
    
    return points;
}
