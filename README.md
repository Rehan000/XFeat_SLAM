# XFeat (Accelerated Features) SLAM

<p align="center">
  <img src="assets/xfeat_slam.gif">
</p>

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
The repo is an attempt to build a monocular SLAM system based on XFeat (Accelerated Features) local features. 
The XFeat model is converted to ONNX format, there are two versions of the models provided with the repo:

:arrow_right: xfeat_vga.onnx (VGA input size (640x480)) <br>
:arrow_right: xfeat.onnx (Input size (1280x1024)) <br>

Torch model is also provided, with the option to select between the ONNX or torch models.

:arrow_right: xfeat.pt <br>


## Installation
The following dependencies must be built in the third-party folder:
```bash
libtorch 2.5.1+cu124
onnxruntime 1.20.0
opencv 4.5.4
```

The following CUDA and cuDNN versions are used:
```bash
CUDA 12.4
cuDNN 9.5.1
```

After the dependencies are added compile and build:
```bash 
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_ONNX=ON
make
```

## Usage
Run the following to start:
```bash 
./xfeat_slam <path_to_model> <path_to_images_directory>
```