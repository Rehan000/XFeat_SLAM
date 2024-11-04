#include "visualize.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat convertToRGB(const cv::Mat& grayscale_image) {
    cv::Mat rgb_image;
    cv::Mat grayscale_8uc1;
    grayscale_image.convertTo(grayscale_8uc1, CV_8UC1);
    cv::cvtColor(grayscale_8uc1, rgb_image, cv::COLOR_GRAY2RGB);
    return rgb_image;
}

void displayFPS(cv::Mat& image, float fps) {
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 3);
}

void drawKeypoints(cv::Mat& image, const std::vector<KeypointData>& keypoint_data, int num_keypoints_to_display) {
    for (size_t i = 0; i < keypoint_data.size(); ++i) {

        for (int j = 0; j < num_keypoints_to_display && j < keypoint_data[i].keypoints.size(0); ++j) {
            // Extract the x and y coordinates of each keypoint
            float x = keypoint_data[i].keypoints[j][0].item<float>();
            float y = keypoint_data[i].keypoints[j][1].item<float>();

            // Draw a circle at each keypoint location
            cv::circle(image, cv::Point(static_cast<int>(x), static_cast<int>(y)), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
}
