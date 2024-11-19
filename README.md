# XFeat SLAM (Accelerated Features for Monocular SLAM)

<p align="center">
  <img src="assets/xfeat_slam.gif" alt="XFeat SLAM in action">
</p>

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Introduction

**XFeat SLAM** is a monocular SLAM system designed around the XFeat (Accelerated Features) local feature detection and matching framework. The system supports feature extraction, pose estimation, and 3D reconstruction, enabling simultaneous localization and mapping with high efficiency and precision.

Key highlights:
- **Modular Architecture**: Designed for extensibility, supporting both ONNX and Torch inference pipelines.
- **Cross-Platform Compatibility**: Runs on CPU or CUDA-enabled GPUs.
- **Real-Time Performance**: Optimized for low-latency processing using modern libraries like ONNX Runtime and LibTorch.

Two versions of the XFeat model are provided:
- `xfeat_vga.onnx`: Optimized for VGA input size (640x480).
- `xfeat.onnx`: Designed for higher-resolution inputs (1280x1024).

Additionally, a TorchScript version of the model is included:
- `xfeat.pt`

---

## Features

1. **Keypoint Detection and Matching**:
   - Detects high-quality, sparse keypoints using XFeat's heatmap-based approach.
   - Performs robust feature matching with cosine similarity and mutual nearest neighbor filtering.

4. **Visualization**:
   - Real-time display of keypoints, matches, and system performance metrics (e.g., FPS).

5. **Model Flexibility**:
   - Seamlessly switch between ONNX and Torch-based inference pipelines based on user requirements.

---

## Installation

### Dependencies

The following dependencies must be built in the `third-party` folder:

```bash
libtorch 2.5.1+cu124
onnxruntime 1.20.0
opencv 4.5.4
```

Ensure you have the following CUDA and cuDNN versions installed:

```bash
CUDA 12.4
cuDNN 9.5.1
```

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/xfeat_slam.git
   cd xfeat_slam
   ```

2. Create a build directory and compile the project:
   ```bash
   mkdir build
   cd build
   cmake .. -DUSE_CUDA=ON -DUSE_ONNX=ON
   make
   ```

---

## Usage

### Running the SLAM System

To run XFeat SLAM on a sequence of images:
```bash
./xfeat_slam <path_to_model> <path_to_images_directory>
```

Example:
```bash
./xfeat_slam ../models/xfeat.onnx ../data/images/
```

### Model Options
Specify the desired model format:
- For ONNX: Provide the `.onnx` model path.
- For TorchScript: Provide the `.pt` model path and ensure the `-DUSE_ONNX=OFF` flag is set during build.

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---