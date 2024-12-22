#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include "xfeat.h"

// Function to convert a grayscale image to RGB
cv::Mat convertToRGB(const cv::Mat& grayscale_image);

// Function to display FPS on an image
void displayFPS(cv::Mat& image, float fps);

// Function to draw keypoints on an RGB image
void drawKeypoints(cv::Mat& image, const std::vector<KeypointData>& keypoint_data, int num_keypoints_to_display);

// Function to initialize a blank trajectory canvas
cv::Mat initializeTrajectoryCanvas(int canvas_size);

// Function to project 3D positions to a 2D canvas
cv::Point projectToCanvas(const Eigen::Vector3d& position, const Eigen::Vector3d& canvas_center, int canvas_size, int scale_factor);

// Function to draw the camera trajectory on the canvas
void drawTrajectory(cv::Mat& canvas, const Eigen::Vector3d& current_position, Eigen::Vector3d& last_position,
                    Eigen::Vector3d& canvas_center, int canvas_size, int scale_factor);

#endif // VISUALIZE_H
