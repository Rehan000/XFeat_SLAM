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
                                  cv::Mat& R, cv::Mat& t,
                                  std::vector<cv::Point2f>& points1_out_filtered,
                                  std::vector<cv::Point2f>& points2_out_filtered,
                                  std::vector<cv::Point2f>& points1_out,
                                  std::vector<cv::Point2f>& points2_out) {

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

    // Mask to store inlier/outlier information
    cv::Mat inlier_mask;

    // Recover pose (rotation and translation)
    int inliers = cv::recoverPose(essential_matrix, points1, points2, camera_intrinsics_, R, t, inlier_mask);

    if (inliers < 8) {
        std::cerr << "Error: Insufficient inliers for reliable pose estimation." << std::endl;
        return false;
    }

    // Filter inlier points based on the inlier mask
    points1_out_filtered.clear();
    points2_out_filtered.clear();
    for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
            points1_out_filtered.push_back(points1[i]);
            points2_out_filtered.push_back(points2[i]);
        }
    }

    // Keypoints
    points1_out = points1;
    points2_out = points2;

    // std::cout << "Pose estimation successful with " << inliers << " inliers." << std::endl;
    return true;
}

// Convert Torch tensors to vector Point2f
std::vector<cv::Point2f> PoseEstimation::convertToPoints(const torch::Tensor& tensor) {
    // Ensure tensor is on CPU and contiguous
    auto tensor_cpu = tensor.to(torch::kCPU).contiguous();

    // Access tensor data and create cv::Mat view
    cv::Mat mat(tensor.size(0), 1, CV_32FC2, (void*)tensor_cpu.data_ptr<float>());
    
    // Convert cv::Mat to vector<cv::Point2f>
    return std::vector<cv::Point2f>(mat.begin<cv::Point2f>(), mat.end<cv::Point2f>());
}

// Convert R and t into transformation matrix
Eigen::Matrix4d PoseEstimation::ConvertToHomogeneous(const cv::Mat& R, const cv::Mat& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T(i, j) = R.at<double>(i, j);
        }
        T(i, 3) = t.at<double>(i, 0);
    }
    return T;
}
