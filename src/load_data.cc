#include "load_data.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

/**
 * @brief Constructor: Initializes the loader and synchronizes the data.
 * @param directory Path to the directory containing metadata files.
 */
RGBDLoader::RGBDLoader(const std::string& directory) {
    std::string rgb_file, depth_file, accelerometer_file, groundtruth_file;

    // Scan directory for required files
    for (const auto& entry : fs::directory_iterator(directory)) {
        const auto& path = entry.path();
        const std::string& filename = path.filename().string();

        if (filename.find("rgb") != std::string::npos) {
            rgb_file = path.string();
        } else if (filename.find("depth") != std::string::npos) {
            depth_file = path.string();
        } else if (filename.find("accelerometer") != std::string::npos) {
            accelerometer_file = path.string();
        } else if (filename.find("groundtruth") != std::string::npos) {
            groundtruth_file = path.string();
        }
    }

    // Check if all files were found
    if (rgb_file.empty() || depth_file.empty() || accelerometer_file.empty() || groundtruth_file.empty()) {
        std::cerr << "Error: Missing required files in the directory." << std::endl;
        std::cerr << "Ensure the directory contains files with 'rgb', 'depth', 'accelerometer', and 'groundtruth' in their names." << std::endl;
        return;
    }

    parseMetadata(rgb_file, rgb_images);
    parseMetadata(depth_file, depth_images);
    parseAccelerometerData(accelerometer_file);
    parseGroundTruthData(groundtruth_file);
    synchronize();
}

void RGBDLoader::parseMetadata(const std::string& file_path, std::map<double, std::string>& image_map) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << file_path << std::endl;
        return;
    }

    fs::path base_directory = fs::path(file_path).parent_path();

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        std::istringstream ss(line);
        double timestamp;
        std::string image_path;
        ss >> timestamp >> image_path;

        fs::path full_path = base_directory / image_path;
        image_map[timestamp] = full_path.string();
    }
}

void RGBDLoader::parseAccelerometerData(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        std::istringstream ss(line);
        double timestamp, ax, ay, az;
        ss >> timestamp >> ax >> ay >> az;
        accelerometer_data[timestamp] = {ax, ay, az};
    }
}

void RGBDLoader::parseGroundTruthData(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        std::istringstream ss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        groundtruth_data[timestamp] = {tx, ty, tz, qx, qy, qz, qw};
    }
}

void RGBDLoader::synchronize() {
    for (const auto& rgb_entry : rgb_images) {
        double rgb_time = rgb_entry.first;

        auto depth_iter = depth_images.lower_bound(rgb_time);
        if (depth_iter == depth_images.end()) continue;

        auto accel_iter = accelerometer_data.lower_bound(rgb_time);
        if (accel_iter != accelerometer_data.end() && std::fabs(rgb_time - accel_iter->first) > 0.03) continue;

        auto gt_iter = groundtruth_data.lower_bound(rgb_time);
        if (gt_iter != groundtruth_data.end() && std::fabs(rgb_time - gt_iter->first) > 0.03) continue;

        synchronized_pairs.emplace_back(
            rgb_entry.second, depth_iter->second, rgb_time,
            accel_iter->second, gt_iter->second
        );
    }

    std::cout << "Synchronized " << synchronized_pairs.size() << " frames." << std::endl;
}

bool RGBDLoader::loadNextFrame(cv::Mat& rgb_image, cv::Mat& depth_image, double& timestamp,
                               std::vector<double>& accel_values, std::vector<double>& groundtruth_values, float depth_scale_factor) {
    if (current_index >= synchronized_pairs.size()) {
        return false;
    }

    const auto& [rgb_path, depth_path, time, accel_data, gt_data] = synchronized_pairs[current_index++];
    timestamp = time;

    rgb_image = cv::imread(rgb_path, cv::IMREAD_COLOR);
    if (rgb_image.empty()) {
        std::cerr << "Error: Could not load RGB image: " << rgb_path << std::endl;
        return false;
    }
    rgb_image.convertTo(rgb_image, CV_32FC3);

    depth_image = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth_image.empty()) {
        std::cerr << "Error: Could not load depth image: " << depth_path << std::endl;
        return false;
    }

    // Convert depth image to meters
    depth_image.convertTo(depth_image, CV_32F, 1.0 / depth_scale_factor);

    accel_values = accel_data;
    groundtruth_values = gt_data;

    return true;
}
