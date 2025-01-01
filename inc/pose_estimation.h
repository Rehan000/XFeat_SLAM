// pose_estimation.h
#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/torch.h> 
#include <Eigen/Dense>

class PoseEstimation {
public:
    // Constructor
    PoseEstimation(const cv::Mat& camera_intrinsics);

    // Estimate pose between two frames
    bool estimatePose(const torch::Tensor& pts_1, 
                      const torch::Tensor& pts_2,
                      cv::Mat& R, cv::Mat& t,
                      std::vector<cv::Point2f>& points1_out_filtered,
                      std::vector<cv::Point2f>& points2_out_filtered,
                      std::vector<cv::Point2f>& points1_out,
                      std::vector<cv::Point2f>& points2_out);

    // Estimate pose using depth information
    bool estimatePoseWithDepth(const cv::Mat& depth1, 
                               const cv::Mat& depth2,
                               const torch::Tensor& pts_1, 
                               const torch::Tensor& pts_2,
                               cv::Mat& R, cv::Mat& t,
                               std::vector<cv::Point2f>& points1_out_filtered,
                               std::vector<cv::Point2f>& points2_out_filtered,
                               std::vector<cv::Point2f>& points1_out,
                               std::vector<cv::Point2f>& points2_out);

    // Convert Torch tensors to vector Point2f
    std::vector<cv::Point2f> convertToPoints(const torch::Tensor& tensor);

    // Convert R and t into transformation matrix
    Eigen::Matrix4d ConvertToHomogeneous(const cv::Mat& R, const cv::Mat& t);

    // Get the global pose
    Eigen::Matrix4d getGlobalPose() const;

private:
    cv::Mat camera_intrinsics_; // Camera intrinsic matrix
    Eigen::Matrix4d global_pose_; // Global pose of the camera
};

#endif // POSE_ESTIMATION_H