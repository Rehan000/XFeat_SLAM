cmake_minimum_required(VERSION 3.16)
project(xfeat_slam)

# Set paths for third-party libraries
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third-party/onnxruntime")
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/third-party/libtorch")

# Option to use ONNX or Torch
option(USE_ONNX "Enable ONNX inference" ON)

# Option to use CUDA
option(USE_CUDA "Enable CUDA support" ON)

# Set CMAKE_PREFIX_PATH to point to libtorch
set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

# Include ONNX Runtime
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include" OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib")
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT}. Please check the path.")
endif()

# Include OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install OpenCV 4.5 or later.")
endif()

# Include libtorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# Include Pangolin
find_package(Pangolin REQUIRED)
if(NOT Pangolin_FOUND)
    message(FATAL_ERROR "Pangolin not found. Install it using your system's package manager or from source.")
endif()

# Check for CUDA if enabled
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "CUDA found, enabling CUDA support.")
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA not found, disabling CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

# Add executable
add_executable(xfeat_slam 
    src/main.cc
    src/model_inference_onnx.cc
    src/model_inference_torch.cc
    src/load_images.cc
    src/interpolate_sparse_2d.cc
    src/xfeat.cc
    src/visualize.cc
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Include ONNX Runtime
target_include_directories(xfeat_slam PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_directories(xfeat_slam PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(xfeat_slam onnxruntime)

# Include OpenCV
target_include_directories(xfeat_slam PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(xfeat_slam ${OpenCV_LIBS})

# Include libtorch
target_link_libraries(xfeat_slam "${TORCH_LIBRARIES}")

# Include Pangolin
target_include_directories(xfeat_slam PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(xfeat_slam ${Pangolin_LIBRARIES})

# Link CUDA libraries if enabled
if(USE_CUDA)
    set(CUDA_LIB_PATH "/usr/local/cuda-12.4/lib64")
    target_link_directories(xfeat_slam PRIVATE ${CUDA_LIB_PATH})
    target_link_libraries(xfeat_slam ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    target_link_libraries(xfeat_slam cudart)
endif()

# Set C++ standard
set_target_properties(xfeat_slam PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
