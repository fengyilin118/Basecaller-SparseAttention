#pragma once
#include <torch/extension.h>

struct QuantizedWeights {
    torch::Tensor weight_int8;     // INT8 weights
    torch::Tensor scales;          // FP32 scales per output channel
    torch::Tensor zero_points;     // INT8 zero points per channel
    bool is_symmetric;             // Whether zero_point is always 0
};

torch::Tensor pruned_qkv_projection_int8(    
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_bias,
    torch::Tensor k_bias,
    torch::Tensor v_bias,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim,
    bool fused,
    bool return_full_size);

torch::Tensor pruned_qkv_projection(
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_bias,
    torch::Tensor k_bias,
    torch::Tensor v_bias,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim,
    bool fused,
    bool return_full_size);

torch::Tensor pruned_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    //torch::Tensor attention_mask,
    float scale);
    
torch::Tensor pruned_output_projection(
    torch::Tensor context,
    torch::Tensor out_weight,
    torch::Tensor out_bias,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim);


