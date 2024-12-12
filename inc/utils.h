#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>

// Function to load the camera intrinsics matrix from a text file
cv::Mat loadCameraIntrinsics(const std::string& file_path);

#endif // UTILS_H
