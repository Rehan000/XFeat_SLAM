// pose_estimation.cc
#include "pose_estimation.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

// Constructor
PoseEstimation::PoseEstimation(const cv::Mat& camera_intrinsics) 
    : camera_intrinsics_(camera_intrinsics.clone()), global_pose_(Eigen::Matrix4d::Identity()) {}

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

    // Update the global pose
    Eigen::Matrix4d T = ConvertToHomogeneous(R, t);
    global_pose_ = global_pose_ * T;

    // std::cout << "Pose estimation successful with " << inliers << " inliers." << std::endl;
    return true;
}

// Estimate relative pose using depth information
bool PoseEstimation::estimatePoseWithDepth(
    const cv::Mat& depth1, const cv::Mat& depth2,
    const torch::Tensor& pts_1, 
    const torch::Tensor& pts_2,
    cv::Mat& R, cv::Mat& t,
    std::vector<cv::Point2f>& points1_out_filtered,
    std::vector<cv::Point2f>& points2_out_filtered,
    std::vector<cv::Point2f>& points1_out,
    std::vector<cv::Point2f>& points2_out) {
    
    std::vector<cv::Point2f> points1 = convertToPoints(pts_1);
    std::vector<cv::Point2f> points2 = convertToPoints(pts_2);

    // Filter valid depth points
    std::vector<cv::Point3f> points3D1;
    std::vector<cv::Point2f> valid_points2;
    for (size_t i = 0; i < points1.size(); ++i) {
        int x = static_cast<int>(points1[i].x);
        int y = static_cast<int>(points1[i].y);

        float depth = depth1.at<float>(y, x);
        if (depth > 0 && depth < 10.0) {
            cv::Point3f point3D(
                (x - camera_intrinsics_.at<double>(0, 2)) * depth / camera_intrinsics_.at<double>(0, 0),
                (y - camera_intrinsics_.at<double>(1, 2)) * depth / camera_intrinsics_.at<double>(1, 1),
                depth);
            points3D1.push_back(point3D);
            valid_points2.push_back(points2[i]);
        }
    }

    if (points3D1.size() < 6 || valid_points2.size() < 6) {
        std::cerr << "Error: Insufficient 3D-2D correspondences for pose estimation." << std::endl;
        return false;
    }

    // Solve PnP problem
    cv::Mat inliers;
    cv::Mat rotation_vector;
    if (!cv::solvePnPRansac(
            points3D1, valid_points2, camera_intrinsics_, cv::Mat(),
            rotation_vector, t, false, 100, 8.0, 0.99, inliers)) {
        std::cerr << "Error: PnP pose estimation failed." << std::endl;
        return false;
    }

    // Convert rotation vector to rotation matrix
    cv::Rodrigues(rotation_vector, R);

    // Filter inlier points based on the inlier mask
    points1_out_filtered.clear();
    points2_out_filtered.clear();
    for (int i = 0; i < inliers.rows; ++i) {
        int idx = inliers.at<int>(i, 0);
        points1_out_filtered.push_back(points1[idx]);
        points2_out_filtered.push_back(points2[idx]);
    }

    // Keypoints
    points1_out = points1;
    points2_out = points2;

    // Update the global pose
    Eigen::Matrix4d T = ConvertToHomogeneous(R, t);
    global_pose_ = global_pose_ * T;

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

// Get the global pose
Eigen::Matrix4d PoseEstimation::getGlobalPose() const {
    return global_pose_;
}