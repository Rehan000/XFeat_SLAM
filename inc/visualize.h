#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "xfeat.h"

// Function to convert a grayscale image to RGB
cv::Mat convertToRGB(const cv::Mat& grayscale_image);

// Function to display FPS on an image
void displayFPS(cv::Mat& image, float fps);

// Function to draw keypoints on an RGB image
void drawKeypoints(cv::Mat& image, const std::vector<KeypointData>& keypoint_data, int num_keypoints_to_display);

#endif // VISUALIZE_H
