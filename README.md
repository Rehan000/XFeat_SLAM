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
:arrow_right: xfeat.onnx (Input size (1280x1024))


## Installation
The following dependencies must be built in the third-party folder:
```bash
libtorch 2.5.0+cu118
onnxruntime 1.19.2 
opencv 4.5.4
```

After the dependencies are added compile and build:
```bash 
mkdir build
cd build
cmake ..
make
```

## Usage
Run the following to start:
```bash 
./xfeat_slam <path_to_model> <path_to_images_directory>
```