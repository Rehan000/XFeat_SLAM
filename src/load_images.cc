#include "load_images.h"
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

// Constructor: Loads the image paths from the directory
ImageLoader::ImageLoader(const std::string& directory, bool is_rgb) 
    : is_rgb_(is_rgb) {
    loadImagePaths(directory);
}

// Function to load images one by one in the specified format, resize, normalize, and return
bool ImageLoader::loadNextImage(cv::Mat& output_image) {
    if (current_image_index >= image_paths.size()) {
        return false; // No more images to load
    }

    // Load the next image
    cv::Mat image;
    if (is_rgb_) {
        image = cv::imread(image_paths[current_image_index], cv::IMREAD_COLOR); // Load as RGB
    } else {
        image = cv::imread(image_paths[current_image_index], cv::IMREAD_GRAYSCALE); // Load as grayscale
    }

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << image_paths[current_image_index] << std::endl;
        return false;
    }

    // Convert the image to float
    if (is_rgb_) {
        image.convertTo(output_image, CV_32FC3); // Convert RGB to float
    } else {
        image.convertTo(output_image, CV_32FC1); // Convert grayscale to float
    }

    // Move to the next image in the sequence
    current_image_index++;

    return true;
}

// Helper function to load all image file paths from the given directory
void ImageLoader::loadImagePaths(const std::string& directory) {
    const std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); // Convert to lowercase

            if (std::find(valid_extensions.begin(), valid_extensions.end(), ext) != valid_extensions.end()) {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    // Sort the image paths to ensure they are in sequence
    std::sort(image_paths.begin(), image_paths.end());
}
