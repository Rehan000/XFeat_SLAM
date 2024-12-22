#ifndef LOAD_IMAGES_H
#define LOAD_IMAGES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

class ImageLoader {
public:
    // Constructor
    ImageLoader(const std::string& directory, bool is_rgb = false);

    // Load the next image
    bool loadNextImage(cv::Mat& output_image);

private:
    // Helper function to load image paths from the directory
    void loadImagePaths(const std::string& directory);

    std::vector<std::string> image_paths; // Stores paths to image files
    size_t current_image_index = 0;       // Tracks the current image index
    bool is_rgb_;                         // Flag to load images in RGB or grayscale
};

#endif // LOAD_IMAGES_H
