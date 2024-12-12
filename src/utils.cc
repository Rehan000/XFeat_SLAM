#include "utils.h"
#include <fstream>
#include <stdexcept>

// Function to load the camera intrinsics matrix from a text file
cv::Mat loadCameraIntrinsics(const std::string& file_path) {
    cv::Mat camera_intrinsics(3, 3, CV_64F);
    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open camera intrinsics file: " + file_path);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!(file >> camera_intrinsics.at<double>(i, j))) {
                throw std::runtime_error("Error: Invalid format in camera intrinsics file.");
            }
        }
    }

    return camera_intrinsics;
}
