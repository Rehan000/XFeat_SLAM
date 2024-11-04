#ifndef LOAD_IMAGES_H
#define LOAD_IMAGES_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

class ImageLoader {
public:
    // Constructor that takes the directory path
    ImageLoader(const std::string& directory);

    // Function to load images one by one in grayscale, resize, normalize, and return
    bool loadNextImage(cv::Mat& output_image);

private:
    std::vector<std::string> image_paths;
    size_t current_image_index = 0;

    // Helper function to load all image file paths from the given directory
    void loadImagePaths(const std::string& directory);
};

#endif
