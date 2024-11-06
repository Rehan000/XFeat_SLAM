#ifndef MODEL_INFERENCE_TORCH_H
#define MODEL_INFERENCE_TORCH_H

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

class ModelInferenceTorch {
public:
    // Constructor to load and initialize the model
    ModelInferenceTorch(const std::string& model_path);

    // Destructor
    ~ModelInferenceTorch();

    // Method to perform inference
    std::vector<at::Tensor> RunInference(const at::Tensor& input);

private:
    torch::jit::script::Module model;
};

#endif // MODEL_INFERENCE_TORCH_H
