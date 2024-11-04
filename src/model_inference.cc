#include "model_inference.h"

// Constructor
ModelInference::ModelInference(const char* model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"), session_options(), session(env, model_path, session_options) {

    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // session_options.SetIntraOpNumThreads(4);
    num_input_nodes = session.GetInputCount();
    num_output_nodes = session.GetOutputCount();

    // Retrieve input names
    input_names.resize(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; ++i) {
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
        input_names[i] = input_name_ptr.get();
    }

    // Retrieve output names
    output_names_str.resize(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; ++i) {
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_names_str[i] = output_name_ptr.get();
    }
}

// Destructor
ModelInference::~ModelInference() {
    // Any necessary cleanup (optional since ONNX Runtime automatically manages memory)
}

// Method to perform inference
std::vector<Ort::Value> ModelInference::RunInference(const std::vector<std::vector<float>>& input_data) {
    std::vector<const char*> input_name_ptrs(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_name_ptrs[i] = input_names[i].c_str();
    }

    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < num_input_nodes; ++i) {
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data[i].data()), input_data[i].size(), input_dims[i].data(), input_dims[i].size())
        );
    }

    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names_str) {
        output_name_ptrs.push_back(name.c_str());
    }

    // Run the model
    return session.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_tensors.data(), num_input_nodes, output_name_ptrs.data(), num_output_nodes);
}



// Print model input/output details
void ModelInference::PrintModelInfo() const {
    for (size_t i = 0; i < num_input_nodes; ++i) {
        std::cout << "Input Name " << i << ": " << input_names[i] << std::endl;
    }
    for (size_t i = 0; i < num_output_nodes; ++i) {
        std::cout << "Output Name " << i << ": " << output_names_str[i] << std::endl;
    }
}
