#include "load_images.h"
#include <iostream>
#include <algorithm> 

namespace fs = std::filesystem;

// Constructor: Loads the image paths from the directory
ImageLoader::ImageLoader(const std::string& directory) {
    loadImagePaths(directory);
}

// Function to load images one by one in grayscale, resize, normalize, and return
bool ImageLoader::loadNextImage(cv::Mat& output_image) {
    if (current_image_index >= image_paths.size()) {
        return false; // No more images to load
    }

    // Load the next image in grayscale format
    cv::Mat image = cv::imread(image_paths[current_image_index], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << image_paths[current_image_index] << std::endl;
        return false;
    }

    // Resize the image to 640x480
    cv::resize(image, image, cv::Size(640, 480));

    // Convert the image to float
    image.convertTo(output_image, CV_32FC1);

    // Move to the next image in the sequence
    current_image_index++;

    return true;
}

// Helper function to load all image file paths from the given directory
void ImageLoader::loadImagePaths(const std::string& directory) {
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            image_paths.push_back(entry.path().string());
        }
    }

    // Sort the image paths to ensure they are in sequence
    std::sort(image_paths.begin(), image_paths.end());
}
 