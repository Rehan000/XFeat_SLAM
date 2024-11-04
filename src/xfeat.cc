#include "xfeat.h"
#include <numeric>
#include <algorithm>

// Constructor for XFeat class
XFeat::XFeat(const std::string& model_path, int top_k, float detection_threshold)
    : top_k(top_k),
      detection_threshold(detection_threshold),
      model_inference(model_path.c_str()),
      device(torch::kCPU)  // Default device initialization to CPU
{
    // Now check if CUDA is available and change device accordingly
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    interpolator = InterpolateSparse2d("bicubic", false, device);
}


// Helper function to print tensor shape
void XFeat::printTensorShape(const torch::Tensor& tensor, const std::string& tensor_name) {
    std::cout << tensor_name << " shape: [";
    for (const auto& size : tensor.sizes()) {
        std::cout << size << " ";
    }
    std::cout << "]" << std::endl;
}

// Helper function to convert Ort::Value tensor to torch::Tensor
torch::Tensor OrtValueToTorchTensor(Ort::Value& ort_value, torch::Device device) {
    torch::InferenceMode guard;

    // Get the shape of the tensor
    Ort::TensorTypeAndShapeInfo shape_info = ort_value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> tensor_shape = shape_info.GetShape();
    
    // Get the number of elements in the tensor
    size_t total_elements = shape_info.GetElementCount();

    // Get a pointer to the raw data in the Ort::Value tensor
    float* ort_data_ptr = ort_value.GetTensorMutableData<float>();

    // Create a torch tensor from the raw data pointer
    torch::Tensor torch_tensor = torch::from_blob(ort_data_ptr, torch::IntArrayRef(tensor_shape), torch::kFloat32);

    // Move the tensor to the appropriate device and clone to ensure it owns the memory
    return torch_tensor.clone().to(device);
}

// Convert logits to heatmap and extract keypoints
torch::Tensor XFeat::getKptsHeatmap(const torch::Tensor& K1, float softmax_temp) {
    torch::InferenceMode guard;

    // Apply softmax to K1 along the channel dimension (dim=1)
    torch::Tensor scores = torch::nn::functional::softmax(K1 * softmax_temp, torch::nn::functional::SoftmaxFuncOptions(1)).narrow(1, 0, 64);

    // Get the dimensions of the scores tensor
    int B = scores.size(0);  // Batch size
    int H = scores.size(2);  // Height
    int W = scores.size(3);  // Width

    // Permute the tensor to (B, H, W, 64)
    torch::Tensor heatmap = scores.permute({0, 2, 3, 1});  // (B, H, W, 64)

    // Reshape the tensor into (B, H, W, 8, 8)
    heatmap = heatmap.reshape({B, H, W, 8, 8});

    // Permute the dimensions to (B, H, 8, W, 8)
    heatmap = heatmap.permute({0, 1, 3, 2, 4});

    // Reshape into final heatmap shape: (B, 1, H*8, W*8)
    heatmap = heatmap.reshape({B, 1, H * 8, W * 8});

    return heatmap;
}

// After generating K1h (heatmap), apply NMS to extract keypoints
torch::Tensor XFeat::NMS(const torch::Tensor& x, float threshold, int kernel_size) {
    torch::InferenceMode guard;

    int B = x.size(0);  // Batch size
    int H = x.size(2);  // Height
    int W = x.size(3);  // Width

    // Perform max pooling to find local maxima
    int pad = kernel_size / 2;
    torch::Tensor local_max = torch::max_pool2d(x, {kernel_size, kernel_size}, {1, 1}, {pad, pad});

    // Find positions where the original value equals the local max and is greater than the threshold
    torch::Tensor mask = (x == local_max) & (x > threshold);

    // Reshape mask to (B, H * W) to apply masked_select
    torch::Tensor mask_flattened = mask.view({B, -1});  // Flatten the H * W dimensions
    torch::Tensor keypoints_flattened = torch::arange(H * W, torch::kLong).unsqueeze(0).repeat({B, 1}).to(mask.device());

    // Select the keypoints that pass the mask (true in the mask)
    torch::Tensor selected_keypoints = torch::masked_select(keypoints_flattened, mask_flattened).view({-1, 1});

    if (selected_keypoints.numel() == 0) {
        return torch::zeros({B, 0, 2}, torch::dtype(torch::kLong).device(x.device()));
    }

    // Convert the flat indices back into (y, x) positions
    torch::Tensor y_ = selected_keypoints / W;
    torch::Tensor x_ = selected_keypoints % W;
    torch::Tensor keypoints = torch::stack({x_, y_}, 1);

    return keypoints.view({B, -1, 2});
}

// Function to detect and compute keypoints and descriptors
std::vector<KeypointData> XFeat::detectAndCompute(const cv::Mat& input_image) {
    torch::InferenceMode guard;

    std::vector<KeypointData> keypoint_data;

    // Convert the input image to a vector of floats
    std::vector<float> input_data;
    input_data.assign(input_image.begin<float>(), input_image.end<float>());
    std::vector<std::vector<float>> input_tensor = {input_data};

    // Run model inference
    auto output_tensors = model_inference.RunInference(input_tensor);

    // M1: Features (1, 64, 60, 80)
    Ort::Value& M1_tensor = output_tensors[0];
    torch::Tensor M1 = OrtValueToTorchTensor(M1_tensor, device);
    M1 = torch::nn::functional::normalize(M1, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // K1: Keypoints (1, 65, 60, 80)
    Ort::Value& K1_tensor = output_tensors[1];
    torch::Tensor K1 = OrtValueToTorchTensor(K1_tensor, device);

    // H1: Heatmap (1, 1, 60, 80)
    Ort::Value& H1_tensor = output_tensors[2];
    torch::Tensor H1 = OrtValueToTorchTensor(H1_tensor, device);

    // Convert logits to heatmap and extract keypoints
    torch::Tensor K1h = getKptsHeatmap(K1);

    // Apply NMS to extract keypoints from the heatmap
    torch::Tensor mkpts = NMS(K1h, detection_threshold, 5);

    // Compute reliability scores
    InterpolateSparse2d _nearest("nearest", false, device);
    InterpolateSparse2d _bilinear("bilinear", false, device);
    int _H1 = input_image.rows;
    int _W1 = input_image.cols;
    torch::Tensor nearest_interpolated = _nearest.forward(K1h, mkpts, _H1, _W1);   // Nearest interpolation on K1h
    torch::Tensor bilinear_interpolated = _bilinear.forward(H1, mkpts, _H1, _W1);  // Bilinear interpolation on H1
    torch::Tensor scores = (nearest_interpolated * bilinear_interpolated).squeeze(-1);
    torch::Tensor mask = torch::all(mkpts == 0, -1);
    mask = mask.view(-1);
    scores.masked_fill_(mask, -1);

    // Select top-k features
    torch::Tensor sorted_idxs = std::get<1>(torch::sort(-scores, -1));
    sorted_idxs = sorted_idxs.squeeze(-2);
    torch::Tensor mkpts_x = torch::gather(mkpts.index({"...", 0}), -1, sorted_idxs).index({"...", torch::indexing::Slice(torch::indexing::None, top_k)});
    torch::Tensor mkpts_y = torch::gather(mkpts.index({"...", 1}), -1, sorted_idxs).index({"...", torch::indexing::Slice(torch::indexing::None, top_k)});
    mkpts = torch::cat({mkpts_x.unsqueeze(-1), mkpts_y.unsqueeze(-1)}, -1);
    sorted_idxs = sorted_idxs.unsqueeze(1);
    scores = torch::gather(scores, -1, sorted_idxs).index({"...", torch::indexing::Slice(torch::indexing::None, top_k)});

    // Interpolate descriptors at kpts positions
    torch::Tensor feats = interpolator.forward(M1, mkpts, _H1, _W1);

    // Apply L2 normalization to the tensor along the specified dimension
    feats =  torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1).p(2));

    // Correct kpt scale
    torch::Tensor scale_tensor = torch::tensor({1, 1}, torch::TensorOptions().device(mkpts.device()));
    scale_tensor = scale_tensor.view({1, 1, -1});
    mkpts = mkpts * scale_tensor;

    // Return keypoints, scores and descriptors
    int B = mkpts.size(0);
    for (int b = 0; b < B; ++b) {
        torch::Tensor valid_mask = scores[b].squeeze(0) > 0;
        torch::Tensor valid_keypoints = mkpts[b].index({valid_mask}).view({-1, 2});
        torch::Tensor valid_scores = scores[b].index({torch::indexing::Slice(), valid_mask}); 
        torch::Tensor valid_descriptors = feats[b].index({torch::indexing::Slice(), valid_mask, torch::indexing::Slice()});
        KeypointData kp_data;
        kp_data.keypoints = valid_keypoints;
        kp_data.scores = valid_scores.squeeze(0);
        kp_data.descriptors = valid_descriptors.squeeze(0);
        keypoint_data.push_back(kp_data);
    }

    return keypoint_data;
}

// Function to perform feature matching between two images and return additional keypoint data
std::tuple<torch::Tensor, torch::Tensor, std::vector<KeypointData>, std::vector<KeypointData>>
XFeat::match_xfeat(const cv::Mat& input_image_1, const cv::Mat& input_image_2, int top_k, float min_cossim) {
    torch::InferenceMode guard;

    if (top_k <= 0) top_k = this->top_k;

    // Detect and compute features for both images
    std::vector<KeypointData> keypoint_data_1 = detectAndCompute(input_image_1);
    std::vector<KeypointData> keypoint_data_2 = detectAndCompute(input_image_2);

    // Ensure keypoints and descriptors are present for matching
    if (keypoint_data_1.empty() || keypoint_data_2.empty() ||
        keypoint_data_1[0].descriptors.numel() == 0 || keypoint_data_2[0].descriptors.numel() == 0) {
        return {
            torch::empty({0, 2}, torch::kFloat32),    // Empty tensor for matched keypoints in image 1
            torch::empty({0, 2}, torch::kFloat32),    // Empty tensor for matched keypoints in image 2
            keypoint_data_1,                          // Return keypoint data for image 1
            keypoint_data_2                           // Return keypoint data for image 2
        };
    }

    // Perform matching between descriptors of the two images
    auto [idxs0, idxs1] = match(keypoint_data_1[0].descriptors, keypoint_data_2[0].descriptors, min_cossim);

    // Return matched keypoints along with the full keypoint data for both images
    return {
        keypoint_data_1[0].keypoints.index_select(0, idxs0),  // Matched keypoints from image 1
        keypoint_data_2[0].keypoints.index_select(0, idxs1),  // Matched keypoints from image 2
        keypoint_data_1,                                      // Full keypoint data from image 1
        keypoint_data_2                                       // Full keypoint data from image 2
    };
}


// Function to perform mutual nearest neighbor (MNN) matching
std::pair<torch::Tensor, torch::Tensor> XFeat::match(const torch::Tensor& feats1, const torch::Tensor& feats2, float min_cossim) {
    torch::InferenceMode guard;

    // Normalize features to use cosine similarity without explicit cosine calculation
    torch::Tensor feats1_norm = torch::nn::functional::normalize(feats1, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
    torch::Tensor feats2_norm = torch::nn::functional::normalize(feats2, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));

    // Compute cosine similarity matrix using normalized features
    torch::Tensor cossim = torch::mm(feats1_norm, feats2_norm.t());

    // Set cosine similarity threshold early to avoid unnecessary processing
    if (min_cossim > 0) {
        cossim = torch::where(cossim > min_cossim, cossim, torch::tensor(-1, torch::kFloat32).to(cossim.device()));
    }

    // Get indices of best matches for each feature in feats1 and feats2
    auto max12 = cossim.max(1);
    torch::Tensor match12 = std::get<1>(max12);  // Indices of best matches for feats1 in feats2
    auto max21 = cossim.transpose(0, 1).max(1);
    torch::Tensor match21 = std::get<1>(max21);  // Indices of best matches for feats2 in feats1

    // Perform mutual nearest neighbor (MNN) check
    torch::Tensor idx0 = torch::arange(match12.size(0), cossim.device());
    torch::Tensor mutual = (match21.index({match12}) == idx0);

    // Filter matches based on MNN and min_cossim criteria
    torch::Tensor idx1;
    if (min_cossim > 0) {
        torch::Tensor good_matches = (std::get<0>(max12) > min_cossim);  // Only retain matches above threshold
        idx0 = idx0.index({mutual & good_matches});
        idx1 = match12.index({mutual & good_matches});
    } else {
        idx0 = idx0.index({mutual});
        idx1 = match12.index({mutual});
    }

    return {idx0, idx1};
}


