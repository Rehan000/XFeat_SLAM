#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

/**
 * @class RGBDLoader
 * @brief Handles the loading and synchronization of RGB, depth, accelerometer, and ground truth data for SLAM pipelines.
 */
class RGBDLoader {
public:
    /**
     * @brief Constructor to initialize the loader with a directory containing metadata files.
     * @param directory Path to the directory containing metadata files (e.g., rgb.txt, depth.txt, accelerometer.txt, groundtruth.txt).
     */
    explicit RGBDLoader(const std::string& directory);

    /**
     * @brief Loads the next synchronized frame (RGB, depth, accelerometer, and ground truth).
     * @param rgb_image Output RGB image.
     * @param depth_image Output depth image.
     * @param timestamp Timestamp of the synchronized frame.
     * @param accel_values Accelerometer values (ax, ay, az).
     * @param groundtruth_values Ground truth pose (tx, ty, tz, qx, qy, qz, qw).
     * @return True if the frame was loaded successfully, false if no more frames are available.
     */
    bool loadNextFrame(cv::Mat& rgb_image, cv::Mat& depth_image, double& timestamp,
                       std::vector<double>& accel_values, std::vector<double>& groundtruth_values, float depth_scale_factor = 5000.0);

private:
    /**
     * @brief Parses the metadata file (e.g., RGB or depth) to extract timestamp-to-filepath mapping.
     * @param file_path Path to the metadata file.
     * @param image_map Map to store the parsed data (timestamp to filepath).
     */
    void parseMetadata(const std::string& file_path, std::map<double, std::string>& image_map);

    /**
     * @brief Parses the accelerometer data file.
     * @param file_path Path to the accelerometer data file.
     */
    void parseAccelerometerData(const std::string& file_path);

    /**
     * @brief Parses the ground truth data file.
     * @param file_path Path to the ground truth data file.
     */
    void parseGroundTruthData(const std::string& file_path);

    /**
     * @brief Synchronizes RGB, depth, accelerometer, and ground truth data based on timestamps.
     */
    void synchronize();

    std::map<double, std::string> rgb_images; ///< Map of timestamps to RGB image file paths.
    std::map<double, std::string> depth_images; ///< Map of timestamps to depth image file paths.
    std::map<double, std::vector<double>> accelerometer_data; ///< Map of timestamps to accelerometer data.
    std::map<double, std::vector<double>> groundtruth_data; ///< Map of timestamps to ground truth data.
    std::vector<std::tuple<std::string, std::string, double, std::vector<double>, std::vector<double>>> synchronized_pairs; ///< Vector of synchronized RGB and depth file paths with timestamps.
    size_t current_index = 0; ///< Current index in the synchronized pairs vector.
};

#endif // LOAD_DATA_H
