#include "model_inference_torch.h"

// Constructor
ModelInferenceTorch::ModelInferenceTorch(const std::string& model_path) {
    try {
        // Load the TorchScript model
        model = torch::jit::load(model_path);
        std::cout << "Model loaded successfully from " << model_path << std::endl;

        // Set the model to evaluation mode
        model.eval();

        // Now check if CUDA is available and change device accordingly
        if (torch::cuda::is_available()) {
            torch::Device device = torch::Device(torch::kCUDA);
            model.to(device);
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }
}

// Destructor
ModelInferenceTorch::~ModelInferenceTorch() {
    // Optional: Cleanup if needed
}

// Method to perform inference
std::vector<at::Tensor> ModelInferenceTorch::RunInference(const at::Tensor& input) {
    std::vector<at::Tensor> outputs;
    try {
        // Run the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        // Forward pass
        auto output = model.forward(inputs).toTuple();
        
        // Extract the output tensors
        outputs.push_back(output->elements()[0].toTensor()); // feats
        outputs.push_back(output->elements()[1].toTensor()); // keypoints
        outputs.push_back(output->elements()[2].toTensor()); // heatmap

    } catch (const c10::Error& e) {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
    }

    return outputs;
}

