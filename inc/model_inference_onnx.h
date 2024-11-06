#ifndef MODEL_INFERENCE_ONNX_H
#define MODEL_INFERENCE_ONNX_H

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

class ModelInferenceONNX {
public:
    // Constructor to load and initialize the model
    ModelInferenceONNX(const char* model_path);

    // Destructor to release resources (if needed)
    ~ModelInferenceONNX();

    // Method to perform inference
    std::vector<Ort::Value> RunInference(const std::vector<std::vector<float>>& input_data);

    // Get the number of inputs and outputs for debugging
    size_t GetNumInputs() const { return num_input_nodes; }
    size_t GetNumOutputs() const { return num_output_nodes; }

    // Print model input/output details for debugging
    void PrintModelInfo() const;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names_str;

    size_t num_input_nodes;
    size_t num_output_nodes;

    std::vector<std::vector<int64_t>> input_dims = {
        // {1, 1, 1024, 1280}  // Input dimensions
        {1, 1, 480, 640} // Input dimensions
    };
};

#endif  // MODEL_INFERENCE_ONNX_H
