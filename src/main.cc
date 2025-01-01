#include "load_images.h"
#include "load_data.h"
#include "xfeat.h"
#include "pose_estimation.h"
#include "utils.h"
#include "frame_manager.h"

// Main Function
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <data_path> <intrinsics_file_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* data_directory = argv[2];
    const char* intrinsics_file_path = argv[3];

    // Create an RGBDLoader object
    RGBDLoader loader(data_directory);

    // Load the camera intrinsics matrix
    cv::Mat camera_intrinsics = loadCameraIntrinsics(intrinsics_file_path);

    // Initialize the feature extraction model
    XFeat xfeat(model_path);

    // Create PoseEstimation object
    PoseEstimation pose_estimator(camera_intrinsics);

    // Create FrameManager object
    FrameManager frame_manager(5); // Max number of frames to be stored

    // Variables for calculating FPS
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float fps = 0.0;

    cv::Mat rgb_image1, rgb_image2;
    cv::Mat depth_image1, depth_image2;
    double timestamp1, timestamp2;
    std::vector<double> accel_values1, accel_values2;
    std::vector<double> groundtruth_values1, groundtruth_values2;

    bool hasNextImage = loader.loadNextFrame(rgb_image1, depth_image1, timestamp1, accel_values1, groundtruth_values1);

    int frame_id = 0;
    while (hasNextImage) {
        std::cout << "Start iteration: " << frame_id << std::endl;

        // Create a new frame
        Frame current_frame(frame_id++, rgb_image1);

        hasNextImage = loader.loadNextFrame(rgb_image2, depth_image2, timestamp2, accel_values2, groundtruth_values2);
        if (!hasNextImage) break;

        start = std::chrono::high_resolution_clock::now();

        // Perform feature matching
        torch::Tensor pts_1, pts_2;
        std::vector<KeypointData> keypoint_data_1, keypoint_data_2;
        std::tie(pts_1, pts_2, keypoint_data_1, keypoint_data_2) = xfeat.match_xfeat(rgb_image1, rgb_image2);

        current_frame.keypoints = keypoint_data_1[0].keypoints;
        current_frame.descriptors = keypoint_data_1[0].descriptors;

        // Pose estimation
        cv::Mat R, t;
        std::vector<cv::Point2f> points1_out_filtered, points2_out_filtered;
        std::vector<cv::Point2f> points1_out, points2_out;

        if (pose_estimator.estimatePoseWithDepth(depth_image1, depth_image2, pts_1, pts_2, R, t, points1_out_filtered, points2_out_filtered, points1_out, points2_out))
        {
            Eigen::Matrix4d global_pose = pose_estimator.getGlobalPose();
            current_frame.pose = global_pose;

            // std::cout << "Global Pose: \n" << global_pose << std::endl;
        }
        
        // Add the frame to the manager
        frame_manager.addFrame(current_frame, points1_out_filtered.size());

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        fps = 1.0f / duration.count();

        std::cout << "FPS: " << fps << std::endl;

        // Prepare for the next iteration
        rgb_image1 = rgb_image2.clone();
    }

    return 0;
}