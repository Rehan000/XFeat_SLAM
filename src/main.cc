#include "xfeat.h"
#include "load_images.h"
#include "visualize.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// Main function
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <images_directory_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* images_directory = argv[2];

    // Create an ImageLoader object
    ImageLoader image_loader(images_directory);

    // Initialize the model
    XFeat xfeat(model_path);

    // Variables for calculating FPS
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float fps = 0.0;

    cv::Mat image1, image2;
    bool hasNextImage = image_loader.loadNextImage(image1);

    while (hasNextImage) {
        hasNextImage = image_loader.loadNextImage(image2);
        if (!hasNextImage) break;

        // Start time measurement
        start = std::chrono::high_resolution_clock::now();

        // Perform image matching and unpack the returned tuple
        torch::Tensor pts_1, pts_2;
        std::vector<KeypointData> keypoint_data_1, keypoint_data_2;
        std::tie(pts_1, pts_2, keypoint_data_1, keypoint_data_2) = xfeat.match_xfeat(image1, image2);
        
        // End time measurement
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        fps = 1.0f / duration.count();  // Calculate FPS
        // std::cout << "FPS: " << fps << std::endl;

        // Convert the grayscale image to RGB for displaying colored keypoints
        cv::Mat rgb_image = convertToRGB(image1);

        // Draw keypoints on the RGB image
        int num_keypoints_to_display = 1000 ;
        drawKeypoints(rgb_image, keypoint_data_1, num_keypoints_to_display);

        // Display FPS on the top-left corner of the RGB image
        displayFPS(rgb_image, fps);

        // Show RGB image with FPS and keypoints
        cv::imshow("Image Frames", rgb_image);
        cv::waitKey(30);  // Display each frame for 30 milliseconds

        image1 = image2.clone();
    }

    return 0;
}

