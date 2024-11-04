cmake_minimum_required(VERSION 3.16)
project(xfeat_slam)

# Set the path to ONNX Runtime, OpenCV, and libtorch
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third-party/onnxruntime")
set(OPENCV_ROOT "${CMAKE_SOURCE_DIR}/third-party/opencv")
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/third-party/libtorch")

# Ensure ONNX Runtime paths are found
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include" OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib")
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT}. Please check the path.")
endif()

# Ensure OpenCV paths are found
if(NOT EXISTS "${OPENCV_ROOT}/include" OR NOT EXISTS "${OPENCV_ROOT}/lib")
    message(FATAL_ERROR "OpenCV library not found in ${OPENCV_ROOT}. Please check the path.")
endif()

# Ensure libtorch paths are found
if(NOT EXISTS "${LIBTORCH_ROOT}/include" OR NOT EXISTS "${LIBTORCH_ROOT}/lib")
    message(FATAL_ERROR "libtorch library not found in ${LIBTORCH_ROOT}. Please check the path.")
endif()

# Set CMAKE_PREFIX_PATH to point to libtorch
set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${OPENCV_ROOT}/include)
include_directories(${LIBTORCH_ROOT}/include)

# Add executable
add_executable(xfeat_slam 
    src/main.cc
    src/model_inference.cc
    src/load_images.cc
    src/interpolate_sparse_2d.cc
    src/xfeat.cc
    src/visualize.cc
)

# Include ONNX Runtime
target_include_directories(xfeat_slam PRIVATE ${ONNXRUNTIME_ROOT}/include)

# Include OpenCV
target_include_directories(xfeat_slam PRIVATE ${OPENCV_ROOT}/include/opencv4)

# Include libtorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# Link ONNX Runtime
target_link_directories(xfeat_slam PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(xfeat_slam onnxruntime)

# Link OpenCV
target_link_directories(xfeat_slam PRIVATE ${OPENCV_ROOT}/lib)
target_link_libraries(xfeat_slam opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# Link libtorch
target_link_libraries(xfeat_slam "${TORCH_LIBRARIES}")

# Set C++ standard
set_target_properties(xfeat_slam PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Ensure the project uses the correct flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

