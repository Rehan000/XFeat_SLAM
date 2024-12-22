#include "frame_manager.h"

// Frame constructor
Frame::Frame(int frame_id, const cv::Mat& img)
    : id(frame_id), image(img), pose(Eigen::Matrix4d::Identity()) {}

// FrameManager constructor
FrameManager::FrameManager(int max_frames)
    : max_frames(max_frames) {}

// Helper function to determine if a frame qualifies as a keyframe
bool FrameManager::isKeyframe(const Frame& current_frame, const Frame& last_keyframe, int num_inliers) {
    // Compute translation distance
    Eigen::Vector3d translation = current_frame.pose.block<3, 1>(0, 3) - last_keyframe.pose.block<3, 1>(0, 3);

    // std::cout << "Translation: " << translation << std::endl;

    double translation_distance = translation.norm();

    // std::cout << "Translation Distance: " << translation_distance << std::endl;

    // Compute rotation angle
    Eigen::Matrix3d R_current = current_frame.pose.block<3, 3>(0, 0);
    Eigen::Matrix3d R_last = last_keyframe.pose.block<3, 3>(0, 0);
    Eigen::Matrix3d R_diff = R_current.transpose() * R_last;
    double rotation_angle = std::acos((R_diff.trace() - 1) / 2.0) * (180.0 / M_PI);

    // std::cout << "Rotation Angle: " << rotation_angle << std::endl;
    // std::cout << "Inliers: " << num_inliers << std::endl;

    // Check thresholds
    return (translation_distance > translation_threshold || rotation_angle > rotation_threshold || num_inliers < min_inliers);
}

// Add a frame to the manager
void FrameManager::addFrame(const Frame& frame, int num_inliers) {
    // Add the frame to recent frames
    frames.push_back(frame);
    if (frames.size() > max_frames) {
        frames.pop_front();
    }

    // Check if it's a keyframe
    if (keyframes.empty() || isKeyframe(frame, keyframes.back(), num_inliers)) {
        keyframes.push_back(frame);
        std::cout << "Keyframe added: " << frame.id << std::endl;
    }
}

// Get the most recent keyframe
const Frame& FrameManager::getLastKeyframe() const {
    if (keyframes.empty()) {
        throw std::runtime_error("No keyframes stored in the manager!");
    }
    return keyframes.back();
}

// Get all stored keyframes
const std::vector<Frame>& FrameManager::getKeyframes() const {
    return keyframes;
}

// Get all stored recent frames
const std::deque<Frame>& FrameManager::getFrames() const {
    return frames;
}
