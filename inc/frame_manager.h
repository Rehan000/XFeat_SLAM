#ifndef FRAME_MANAGER_H
#define FRAME_MANAGER_H

#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <torch/torch.h>
#include <iostream>

// Frame class to represent individual frames in the SLAM pipeline
class Frame {
public:
    int id; // Unique identifier for the frame
    cv::Mat image; // Image associated with the frame
    Eigen::Matrix4d pose; // Camera pose (4x4 transformation matrix)
    torch::Tensor keypoints; // N x 2 tensor for keypoint coordinates
    torch::Tensor descriptors; // N x D tensor for feature descriptors (e.g., D = 64 for xfeat)

    Frame(int frame_id, const cv::Mat& img);
};

// FrameManager class to handle frame and keyframe storage
class FrameManager {
private:
    std::deque<Frame> frames; // Stores recent frames
    std::vector<Frame> keyframes; // Stores selected keyframes
    int max_frames; // Maximum number of recent frames to store

    // Thresholds for keyframe selection
    const double translation_threshold = 0.5; // Adjust as needed
    const double rotation_threshold = 10.0;  // In degrees
    const int min_inliers = 30;              // Minimum number of matches

    // Helper function to determine if a frame qualifies as a keyframe
    bool isKeyframe(const Frame& current_frame, const Frame& last_keyframe, int num_inliers);

public:
    FrameManager(int max_frames = 5);

    // Add a frame to the manager
    void addFrame(const Frame& frame, int num_inliers);

    // Get the most recent keyframe
    const Frame& getLastKeyframe() const;

    // Get all stored keyframes
    const std::vector<Frame>& getKeyframes() const;

    // Get all recent frames
    const std::deque<Frame>& getFrames() const;
};

#endif // FRAME_MANAGER_H
