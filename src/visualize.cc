#include "visualize.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Function to convert a grayscale image to RGB
cv::Mat convertToRGB(const cv::Mat& grayscale_image) {
    cv::Mat rgb_image;
    cv::Mat grayscale_8uc1;
    grayscale_image.convertTo(grayscale_8uc1, CV_8UC1);
    cv::cvtColor(grayscale_8uc1, rgb_image, cv::COLOR_GRAY2RGB);
    return rgb_image;
}

// Function to display FPS on an image
void displayFPS(cv::Mat& image, float fps) {
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 3);
}

// Function to initialize a blank trajectory canvas
cv::Mat initializeTrajectoryCanvas(int canvas_size) {
    return cv::Mat::zeros(canvas_size, canvas_size, CV_8UC3);
}

// Function to update the canvas center dynamically based on the camera position
void updateCanvasCenter(Eigen::Vector3d& canvas_center, const Eigen::Vector3d& current_position, int canvas_size, int scale_factor) {
    const int margin = canvas_size / 5; // 20% margin to trigger panning

    // Project the current position relative to the canvas center
    double x_pos = (current_position.x() - canvas_center.x()) * scale_factor + canvas_size / 2;
    double z_pos = (current_position.z() - canvas_center.z()) * scale_factor + canvas_size / 2;

    // Adjust the canvas center if the position is near the edges
    if (x_pos < margin) canvas_center.x() -= (margin - x_pos) / scale_factor;
    if (x_pos > canvas_size - margin) canvas_center.x() += (x_pos - (canvas_size - margin)) / scale_factor;
    if (z_pos < margin) canvas_center.z() -= (margin - z_pos) / scale_factor;
    if (z_pos > canvas_size - margin) canvas_center.z() += (z_pos - (canvas_size - margin)) / scale_factor;
}

// Function to project 3D positions to a 2D canvas (top-down view)
cv::Point projectToCanvas(const Eigen::Vector3d& position, const Eigen::Vector3d& canvas_center, int canvas_size, int scale_factor) {
    int x = static_cast<int>((position.x() - canvas_center.x()) * scale_factor + canvas_size / 2);
    int y = static_cast<int>((position.z() - canvas_center.z()) * scale_factor + canvas_size / 2); // Use Z directly for vertical positioning
    return cv::Point(x, y);
}

// Function to draw the camera trajectory on the canvas
void drawTrajectory(cv::Mat& canvas, const Eigen::Vector3d& current_position, Eigen::Vector3d& last_position,
                    Eigen::Vector3d& canvas_center, int canvas_size, int scale_factor) {
    // Update the canvas center dynamically
    updateCanvasCenter(canvas_center, current_position, canvas_size, scale_factor);

    // Convert positions to 2D canvas points
    cv::Point current_point = projectToCanvas(current_position, canvas_center, canvas_size, scale_factor);
    cv::Point last_point = projectToCanvas(last_position, canvas_center, canvas_size, scale_factor);

    // Draw trajectory
    if (last_position != Eigen::Vector3d(0, 0, 0)) { // Skip the first frame
        cv::line(canvas, last_point, current_point, cv::Scalar(0, 255, 0), 2); // Green trajectory line
    }
    cv::circle(canvas, current_point, 3, cv::Scalar(0, 0, 255), -1); // Current position as red dot

    // Update the last position
    last_position = current_position;
}
