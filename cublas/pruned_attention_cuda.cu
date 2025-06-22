#include "pruned_attention.h"
#include "tensor_core_optimizer.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>


#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Error checking macro for cuBLAS calls
#define CHECK_CUBLAS(func)                                                \
{                                                                         \
    cublasStatus_t status = (func);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__);            \
        exit(EXIT_FAILURE) ;                                        \
    }                                                                     \
}


#define CHECK_CUBLASLT(func)                                                  \
{                                                                            \
    cublasStatus_t status = (func);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        printf("cuBLASLt error at %s:%d - Status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE);                                                 \
    }                                                                        \
}

// Benchmark function
template<typename Func>
double benchmark(Func func, int warmup_iters, int bench_iters) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        func();
    }
    
    // Benchmark
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < bench_iters; i++) {
        func();
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / (double)bench_iters / 1000.0; // ms per iteration
}

// Helper function to get dense matrices for remaining heads only
std::vector<torch::Tensor> extract_pruned_weights(
    const torch::Tensor& orig_weight,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim,
    bool is_output = false) {
    
    std::vector<torch::Tensor> pruned_weights;
    int num_remaining_heads = head_indices.size();
    
    // Handle separately for q,k,v weights vs output weights
    if (!is_output) {
        // For q,k,v: shape is [hidden_size, embed_dim]
        // We need to extract rows corresponding to kept heads
        auto weight_per_head = orig_weight.view({orig_num_heads, head_dim, -1});
        auto pruned_weight = torch::zeros({num_remaining_heads, head_dim, weight_per_head.size(2)}, 
                                         weight_per_head.options());
        
        // Copy only rows for remaining heads
        for (int i = 0; i < num_remaining_heads; i++) {
            int head_idx = head_indices[i];
            pruned_weight[i].copy_(weight_per_head[head_idx]);
        }
        
        // Reshape back to 2D
        pruned_weight = pruned_weight.reshape({num_remaining_heads * head_dim, weight_per_head.size(2)});
        pruned_weights.push_back(pruned_weight.contiguous());
    } else {
        // For output: shape is [embed_dim, hidden_size]
        // We need to extract columns corresponding to kept heads
        auto weight_per_head = orig_weight.view({-1, orig_num_heads, head_dim});
        auto pruned_weight = torch::zeros({weight_per_head.size(0), num_remaining_heads, head_dim}, 
                                         weight_per_head.options());
        
        // Copy only columns for remaining heads
        for (int i = 0; i < num_remaining_heads; i++) {
            int head_idx = head_indices[i];
            pruned_weight.select(1, i).copy_(weight_per_head.select(1, head_idx));
        }
        
        // Reshape back to 2D
        pruned_weight = pruned_weight.reshape({weight_per_head.size(0), num_remaining_heads * head_dim});
        pruned_weights.push_back(pruned_weight);
    }
    
    return pruned_weights;
}

torch::Tensor extract_pruned_bias(
    const torch::Tensor& orig_bias,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim) {
    
    // If bias is undefined (None in Python), return undefined tensor
    if (!orig_bias.defined()) {
        return torch::Tensor();
    }
    
    // Number of remaining heads
    int num_remaining_heads = head_indices.size();
    
    // Reshape bias to per-head form: [orig_num_heads, head_dim]
    auto bias_per_head = orig_bias.view({orig_num_heads, head_dim});
    
    // Create tensor for pruned bias
    auto pruned_bias = torch::zeros({num_remaining_heads, head_dim}, bias_per_head.options());
    
    // Copy only biases for remaining heads
    for (int i = 0; i < num_remaining_heads; i++) {
        int head_idx = head_indices[i];
        pruned_bias[i].copy_(bias_per_head[head_idx]);
    }
    
    // Reshape back to 1D vector
    pruned_bias = pruned_bias.reshape({num_remaining_heads * head_dim});
    
    return pruned_bias;
}

QuantizedWeights quantize_weights_per_channel(torch::Tensor fp32_weights) {
    // fp32_weights shape: [out_channels, in_channels] [2304, 768]
    int out_channels = fp32_weights.size(0);
    int in_channels = fp32_weights.size(1);

    auto scales = torch::empty({out_channels}, fp32_weights.options());
    auto zero_points = torch::zeros({out_channels}, torch::kInt8);
    auto weight_int8 = torch::empty_like(fp32_weights, torch::kInt8);
    
    // Quantize each output channel separately
    for (int ch = 0; ch < out_channels; ch++) {
        auto channel_weights = fp32_weights[ch];  // [in_channels]
        
        // Calculate min/max for this channel
        float min_val = channel_weights.min().item<float>();
        float max_val = channel_weights.max().item<float>();
        
        // Symmetric quantization (zero_point = 0)
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        float scale = abs_max / 127.0f;  // INT8 range: -128 to 127
        scales[ch] = scale;
        
        // Quantize: W_q = round(W_f / scale)
        auto quantized_channel = (channel_weights / scale).round().clamp(-128, 127);
        weight_int8[ch] = quantized_channel.to(torch::kInt8);
    }
    
    return {weight_int8, scales, zero_points, true};
}

torch::Tensor fp16_gemm(
    torch::Tensor input,                    // FP32 input [batch*seq, in_features]
    torch::Tensor weight,                   // FP16 weight [out_features, in_features]
    ) {
    
    int batch_seq = input.size(0);      // 91776
    int embed_dim = input.size(1);      // 768  
    int hidden_size = weight.size(0);   // 2304

    
    auto weight_contiguous = weight.contiguous();

    // Step 2: Convert input to FP16
    auto input_fp16 = input.to(torch::kFloat16).contiguous();
    auto weight_fp16 = weight.to(torch::kFloat16).contiguous();
        
    // Create output tensor for fused QKV
    auto fused_out = torch::empty({batch_seq, hidden_size}, input_fp16.options());

    // Step 3: Prepare output tensor for FP32 accumulation
    auto output_fp32 = torch::zeros({batch_seq, hidden_size}, torch::kFloat32).cuda();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_T, 
            CUBLAS_OP_N,
            hidden_size,
            batch_seq, 
            embed_dim, 
            &alpha,
            weight_fp16.data_ptr(), 
            CUDA_R_16F, 
            embed_dim,
            hidden_reshaped_fp16.data_ptr(),    
            CUDA_R_16F, 
            embed_dim,
            &beta,
            fused_out.data_ptr(),          
            CUDA_R_16F, 
            3*reduced_hidden_size,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP    // Let cuBLAS choose best algorithm for RTX3060
        ));
        CHECK_CUDA(cudaDeviceSynchronize());      // Force synchronization
        printf("FP16 operation completed successfully\n");

        
        fused_out=fused_out.to(torch::kFloat32);
        return fused_out;  // Return the fused output tensor
    
    }

torch::Tensor int8_quantized_gemm_optimized(
    torch::Tensor input,                    // FP32 input [batch*seq, in_features]
    const QuantizedWeights& quantized_weights  // Pre-quantized INT8 weights
    ) {   
    
    
    int batch_seq = input.size(0);      // 91776
    int embed_dim = input.size(1);      // 768  
    int hidden_size = quantized_weights.weight_int8.size(0);  // 2304


    auto weights_int8 = quantized_weights.weight_int8.contiguous();

    // Make contiguous and quantize
    auto input_contiguous = input.contiguous();
    auto weight_contiguous = quantized_weights.weight_int8.contiguous();

    // Check if weight is actually INT8
    if (weight_contiguous.dtype() != torch::kInt8) {
        printf("ERROR: Weight is not INT8! dtype: %s\n", 
           torch::toString(weight_contiguous.dtype()).c_str());
    }

    // Step 2: Quantize input to INT8
    float input_scale = input_contiguous.abs().max().item<float>() / 127.0f;
    auto input_int8 = (input_contiguous / input_scale).round().clamp(-128, 127).to(torch::kInt8);
    input_int8 = input_int8.contiguous();

  // Step 3: Debug tensor info
  /*
    printf("\nTensor info:\n");
    printf("Input INT8 shape: [%ld, %ld], stride: [%ld, %ld]\n", 
        input_int8.size(0), input_int8.size(1), 
        input_int8.stride(0), input_int8.stride(1));
    printf("Weight INT8 shape: [%ld, %ld], stride: [%ld, %ld]\n", 
       weight_contiguous.size(0), weight_contiguous.size(1),
       weight_contiguous.stride(0), weight_contiguous.stride(1));
*/
    auto output_int32 = torch::zeros({batch_seq, hidden_size}, torch::kInt32).cuda();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    int32_t alpha = 1;
    int32_t beta = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                        // A: weight, no transpose
        CUBLAS_OP_N,                        // B: input, transpose
        hidden_size,                          // m: 2304
        batch_seq,                        // n: 91176
        embed_dim,                          // k: 768
        &alpha,                             // alpha
        weight_contiguous.data_ptr<int8_t>(),              // A matrix  
        CUDA_R_8I,                           // A type
        embed_dim,                          // lda: 768
        input_int8.data_ptr<int8_t>(),             // B matrix
        CUDA_R_8I,                           // B type  
        embed_dim,                          // ldb: 768
        &beta,                              // beta
        output_int32.data_ptr<int32_t>(),             // C matrix
        CUDA_R_32I,                          // C type
        hidden_size,                          // ldc: 2304
        CUBLAS_COMPUTE_32I,                         // Compute type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP       // Algorithm
    ));
    CHECK_CUDA(cudaDeviceSynchronize());  
    printf("cuBLAS INT8 GEMM completed successfully\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("cuBLAS INT8 GEMM took %ld microseconds\n", duration.count()/ 1000); // ms per iteration
 printf("cuBLAS INT8 GEMM took %f ms\n", duration.count() / 1000.0);      
    
    // Step 4: Dequantize with proper scaling
    auto output_fp32 = output_int32.to(torch::kFloat32).contiguous();
   

   // Apply scaling
    if (quantized_weights.scales.size(0) == hidden_size) {
        auto combined_scales = input_scale * quantized_weights.scales.unsqueeze(0);
        output_fp32 = output_fp32 * combined_scales;
          
    } else if (quantized_weights.scales.size(0) == 1) {
        float combined_scale = input_scale * quantized_weights.scales.item<float>();
        output_fp32 = output_fp32 * combined_scale;
        
    } else {
        printf("WARNING: Unexpected scales shape: %ld\n", quantized_weights.scales.size(0));
        output_fp32 = output_fp32 * input_scale;
      
    }

    return output_fp32;
    
}

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
    bool return_full_size)
{
   
    // Ensure input is on CUDA
    TORCH_CHECK(hidden_states.device().is_cuda(), "hidden_states must be a CUDA tensor");
    
    // Get tensor dimensions
    int batch_size = hidden_states.size(0);
    int seq_len = hidden_states.size(1);
    int embed_dim = hidden_states.size(2);
    int num_remaining_heads = head_indices.size();

    // Get cuBLAS handle
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Set cuBLAS stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Reshape hidden states for GEMM
    auto hidden_reshaped = hidden_states.reshape({batch_size * seq_len, embed_dim}).contiguous();
    
    // Extract pruned weights for q, k, v projections
    auto q_weights = extract_pruned_weights(q_weight, head_indices, orig_num_heads, head_dim);
    auto k_weights = extract_pruned_weights(k_weight, head_indices, orig_num_heads, head_dim);
    auto v_weights = extract_pruned_weights(v_weight, head_indices, orig_num_heads, head_dim);


    // Extract pruned biases if they exist
    torch::Tensor q_pruned_bias, k_pruned_bias, v_pruned_bias;
    bool has_bias = q_bias.defined();
    
    if (has_bias) {
        q_pruned_bias = extract_pruned_bias(q_bias, head_indices, orig_num_heads, head_dim);
        k_pruned_bias = extract_pruned_bias(k_bias, head_indices, orig_num_heads, head_dim);
        v_pruned_bias = extract_pruned_bias(v_bias, head_indices, orig_num_heads, head_dim);
    }


    torch::Tensor q_out, k_out, v_out;
    
    if (fused && hidden_states.dtype() == torch::kFloat) {
        // FUSED QKV PROJECTION (for high sparsity layers)
        // Create a single fused weight matrix [3 * reduced_hidden_size, embed_dim]
        int reduced_hidden_size = num_remaining_heads * head_dim;
        auto fused_weight = torch::cat({q_weights[0], k_weights[0], v_weights[0]}, 0);
        auto fused_bias = torch::cat({q_pruned_bias, k_pruned_bias, v_pruned_bias}, 0);

        auto hidden_reshaped_int8=hidden_reshaped.clone().detach();
        auto quantized_weights = quantize_weights_per_channel(fused_weight);

        auto fused_out = int8_quantized_gemm_optimized(hidden_reshaped, quantized_weights);
        printf("return successfully\n");
    
        
        
/*
    
        */
        // Add bias
        fused_out = fused_out.transpose(0, 1);  // [3*reduced_hidden, batch*seq]
        fused_out = fused_out.add(fused_bias.unsqueeze(1));
        fused_out = fused_out.transpose(0, 1);  // [batch*seq, 3*reduced_hidden]
        
        // Split fused output back into q, k, v
        auto outputs = fused_out.chunk(3, 1);
        q_out = outputs[0];
        k_out = outputs[1];
        v_out = outputs[2];
    } 
    
    torch::Tensor q_out_final, k_out_final, v_out_final;
    
    if (return_full_size) {
        // Create FULL SIZE output tensors with zeros
        q_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        k_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        v_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        
        // Reshape temporary outputs
        q_out = q_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        k_out = k_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        v_out = v_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        
        // Copy computed values to correct positions in full output
        for (int i = 0; i < num_remaining_heads; i++) {
            int head_idx = head_indices[i];
            q_out_final.select(2, head_idx).copy_(q_out.select(2, i));
            k_out_final.select(2, head_idx).copy_(k_out.select(2, i));
            v_out_final.select(2, head_idx).copy_(v_out.select(2, i));
        }
        
        // Pruned heads remain zero by default
        
    } else {
        // Return compact tensors (only remaining heads)
        q_out_final = q_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        k_out_final = k_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        v_out_final = v_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
    }
    // Stack and return
    return torch::stack({q_out_final, k_out_final, v_out_final}, 0);
}

// Implementation for pruned QKV projection

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
    bool return_full_size) {
    
    // Ensure input is on CUDA
    TORCH_CHECK(hidden_states.device().is_cuda(), "hidden_states must be a CUDA tensor");
    
    // Get tensor dimensions
    int batch_size = hidden_states.size(0);
    int seq_len = hidden_states.size(1);
    int embed_dim = hidden_states.size(2);
    int num_remaining_heads = head_indices.size();

    // Get cuBLAS handle
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Set cuBLAS stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Reshape hidden states for GEMM
    auto hidden_reshaped = hidden_states.reshape({batch_size * seq_len, embed_dim}).contiguous();
    
    // Extract pruned weights for q, k, v projections
    auto q_weights = extract_pruned_weights(q_weight, head_indices, orig_num_heads, head_dim);
    auto k_weights = extract_pruned_weights(k_weight, head_indices, orig_num_heads, head_dim);
    auto v_weights = extract_pruned_weights(v_weight, head_indices, orig_num_heads, head_dim);

    // Print debug info before the GEMM call
    /*
    fprintf(stderr, "GEMM dimensions: m=%d, n=%d, k=%d\n", 
        batch_size * seq_len, num_remaining_heads * head_dim, embed_dim);
    fprintf(stderr, "GEMM leading dimensions: lda=%d, ldb=%d, ldc=%d\n", 
        embed_dim, embed_dim, num_remaining_heads * head_dim);
    fprintf(stderr, "Matrix A (hidden) shape: [%ld, %ld]\n", 
        hidden_reshaped.size(0), hidden_reshaped.size(1));
    fprintf(stderr, "Matrix B (q_weights) shape: [%ld, %ld]\n", 
        q_weights[0].size(0), q_weights[0].size(1));
    */

    // Extract pruned biases if they exist
    torch::Tensor q_pruned_bias, k_pruned_bias, v_pruned_bias;
    bool has_bias = q_bias.defined();
    
    if (has_bias) {
        q_pruned_bias = extract_pruned_bias(q_bias, head_indices, orig_num_heads, head_dim);
        k_pruned_bias = extract_pruned_bias(k_bias, head_indices, orig_num_heads, head_dim);
        v_pruned_bias = extract_pruned_bias(v_bias, head_indices, orig_num_heads, head_dim);
    }


    torch::Tensor q_out, k_out, v_out;
    
    if (fused && hidden_states.dtype() == torch::kFloat) {
        // FUSED QKV PROJECTION (for high sparsity layers)
        // Create a single fused weight matrix [3 * reduced_hidden_size, embed_dim]
        int reduced_hidden_size = num_remaining_heads * head_dim;
        auto fused_weight = torch::cat({q_weights[0], k_weights[0], v_weights[0]}, 0);
        auto fused_bias = torch::cat({q_pruned_bias, k_pruned_bias, v_pruned_bias}, 0);
        
        // Create output tensor for fused QKV
        auto fused_out = torch::empty({batch_size * seq_len, 3 * reduced_hidden_size}, 
                                     hidden_states.options());
        
        /*
        RTX3060Optimizer optimizer;
        int m = 3*reduced_hidden_size;
        int n = batch_size * seq_len;
        int k = embed_dim;
        
        auto [m_opt, n_opt, k_opt] = optimizer.get_memory_efficient_dims(m, n, k, torch::kFloat16);
        int optimal_batch_size = optimizer.get_optimal_batch_size(batch_size);

        printf("Optimal batch size: %d\n", optimal_batch_size);
        printf("RTX 3060 optimized dims: [%d, %d, %d] -> [%d, %d, %d]\n", m, n, k, m_opt, n_opt, k_opt);
        */

        // Perform single GEMM for fused QKV projection
        float alpha = 1.0f;
        float beta = 0.0f;
        /*
        CHECK_CUBLAS(cublasSgemm_v2(
            handle,
            CUBLAS_OP_T,              
            CUBLAS_OP_N,              
            3 * reduced_hidden_size,   //M
            batch_size * seq_len,     //N
            embed_dim,     // K
            &alpha,                   // scaling factor
            fused_weight.data_ptr<float>(),  // fused weight matrix
            embed_dim,                // leading dimension of A
            hidden_reshaped.data_ptr<float>(),  // input matrix
            embed_dim,               // leading dimension of B
            &beta,                    // scaling factor for C
            fused_out.data_ptr<float>(),  // output matrix
            3 * reduced_hidden_size   // leading dimension of C
        ));*/
        
       CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_T, 
            CUBLAS_OP_N,
            3*reduced_hidden_size,
            batch_size * seq_len, 
            embed_dim,                    // Your dimensions are already RTX3060-optimal
            &alpha,
            fused_weight.data_ptr(), 
            CUDA_R_16F, 
            embed_dim,
            hidden_reshaped.data_ptr(),    
            CUDA_R_16F, 
            embed_dim,
            &beta,
            fused_out.data_ptr(),          
            CUDA_R_16F, 
            3*reduced_hidden_size,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP    // Let cuBLAS choose best algorithm for RTX3060
        ));
        fused_out = fused_out.to(torch::kFloat32).contiguous();
        // Add bias
        fused_out = fused_out.transpose(0, 1);  // [3*reduced_hidden, batch*seq]
        fused_out = fused_out.add(fused_bias.unsqueeze(1));
        fused_out = fused_out.transpose(0, 1);  // [batch*seq, 3*reduced_hidden]
        
        // Split fused output back into q, k, v
        auto outputs = fused_out.chunk(3, 1);
        q_out = outputs[0];
        k_out = outputs[1];
        v_out = outputs[2];
    } else {
        // SEPARATE Q, K, V PROJECTIONS
        // Create output tensors
        q_out = torch::empty({batch_size * seq_len, num_remaining_heads * head_dim}, 
                           hidden_states.options());
        k_out = torch::empty({batch_size * seq_len, num_remaining_heads * head_dim}, 
                           hidden_states.options());
        v_out = torch::empty({batch_size * seq_len, num_remaining_heads * head_dim}, 
                           hidden_states.options());
        
        // Perform separate GEMMs for Q, K, V projections
        if (hidden_states.dtype() == torch::kFloat) {
            float alpha = 1.0f;
            float beta = 0.0f;
            
            // Q projection
            CHECK_CUBLAS(cublasSgemm_v2(
                handle,
                CUBLAS_OP_T,                      // Transpose for input (A)
                CUBLAS_OP_N,                      // No transpose for weight (B)
                num_remaining_heads * head_dim,   // m: rows of output/B
                batch_size * seq_len,             // n: cols of output/A^T
                embed_dim,                        // k: cols of B, rows of A
                &alpha,                           // scaling factor
                q_weights[0].data_ptr<float>(),   // B matrix (weight) now first
                embed_dim,                        // ldb: leading dimension of B
                hidden_reshaped.data_ptr<float>(), // A matrix (input) now second
                embed_dim,                        // lda: leading dimension of A
                &beta,                            // scaling factor for C
                q_out.data_ptr<float>(),          // C matrix (output)
                num_remaining_heads * head_dim    // ldc: leading dimension of C
            ));
            
            // K projection
            CHECK_CUBLAS(cublasSgemm_v2(
                handle,
                CUBLAS_OP_T,                      // Transpose for input (A)
                CUBLAS_OP_N,                      // No transpose for weight (B)
                num_remaining_heads * head_dim,   // m: rows of output/B
                batch_size * seq_len,             // n: cols of output/A^T
                embed_dim,                        // k: cols of B, rows of A
                &alpha,                           // scaling factor
                k_weights[0].data_ptr<float>(),   // B matrix (weight) now first
                embed_dim,                        // ldb: leading dimension of B
                hidden_reshaped.data_ptr<float>(), // A matrix (input) now second
                embed_dim,                        // lda: leading dimension of A
                &beta,                            // scaling factor for C
                k_out.data_ptr<float>(),          // C matrix (output)
                num_remaining_heads * head_dim    // ldc: leading dimension of C
            ));

            // V projection
            CHECK_CUBLAS(cublasSgemm_v2(
                handle,
                CUBLAS_OP_T,                      // Transpose for input (A)
                CUBLAS_OP_N,                      // No transpose for weight (B)
                num_remaining_heads * head_dim,   // m: rows of output/B
                batch_size * seq_len,             // n: cols of output/A^T
                embed_dim,                        // k: cols of B, rows of A
                &alpha,                           // scaling factor
                v_weights[0].data_ptr<float>(),   // B matrix (weight) now first
                embed_dim,                        // ldb: leading dimension of B
                hidden_reshaped.data_ptr<float>(), // A matrix (input) now second
                embed_dim,                        // lda: leading dimension of A
                &beta,                            // scaling factor for C
                v_out.data_ptr<float>(),          // C matrix (output)
                num_remaining_heads * head_dim    // ldc: leading dimension of C
            ));

        } else {
            // Handle other data types (half, bfloat16, etc.) if needed
            AT_ERROR("Unsupported data type for pruned_qkv_projection");
        }
        
        // Add biases
        if (q_pruned_bias.defined()) {
            q_out = q_out.add(q_pruned_bias);
            k_out = k_out.add(k_pruned_bias);
            v_out = v_out.add(v_pruned_bias);
        }
    }
    
    torch::Tensor q_out_final, k_out_final, v_out_final;
    
    if (return_full_size) {
        // Create FULL SIZE output tensors with zeros
        q_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        k_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        v_out_final = torch::zeros({batch_size, seq_len, orig_num_heads, head_dim}, 
                                  hidden_states.options());
        
        // Reshape temporary outputs
        q_out = q_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        k_out = k_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        v_out = v_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        
        // Copy computed values to correct positions in full output
        for (int i = 0; i < num_remaining_heads; i++) {
            int head_idx = head_indices[i];
            q_out_final.select(2, head_idx).copy_(q_out.select(2, i));
            k_out_final.select(2, head_idx).copy_(k_out.select(2, i));
            v_out_final.select(2, head_idx).copy_(v_out.select(2, i));
        }
        
        // Pruned heads remain zero by default
        
    } else {
        // Return compact tensors (only remaining heads)
        q_out_final = q_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        k_out_final = k_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
        v_out_final = v_out.view({batch_size, seq_len, num_remaining_heads, head_dim});
    }
    // Stack and return
    return torch::stack({q_out_final, k_out_final, v_out_final}, 0);
}


// Implement the attention computation part
torch::Tensor pruned_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    //torch::Tensor mask,
    float scale) {
    
    // Ensure inputs are on CUDA
    TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.device().is_cuda(), "value must be a CUDA tensor");
    
    // Get tensor dimensions
    int batch_size = query.size(0);
    int num_heads = query.size(2);
    int seq_len_q = query.size(1);
    int seq_len_k = key.size(1);
    int head_dim = query.size(3);
    
    // Reshape for efficient matrix multiplication
    auto q_reshaped = query.permute({0, 2, 1, 3})  // [batch, heads, seq_q, dim]
                           .contiguous()
                           .view({batch_size * num_heads, seq_len_q, head_dim});
    
    auto k_reshaped = key.permute({0, 2, 1, 3})    // [batch, heads, seq_k, dim]
                         .contiguous()
                         .view({batch_size * num_heads, seq_len_k, head_dim});
    
    auto v_reshaped = value.permute({0, 2, 1, 3})  // [batch, heads, seq_v, dim]
                           .contiguous()
                           .view({batch_size * num_heads, seq_len_k, head_dim});
    
    // Create output tensor for attention scores
    auto attention_scores = torch::empty({batch_size * num_heads, seq_len_q, seq_len_k}, 
                                       query.options());
    
    // Get cuBLAS handle
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Set cuBLAS stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Calculate attention scores: Q * K^T
    if (query.dtype() == torch::kFloat) {
        float alpha = scale;  // scale = 1/sqrt(head_dim)
        float beta = 0.0f;
        
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_T,       // Transpose key
            CUBLAS_OP_N,       // No transpose for query
            seq_len_k,         // m: rows of op(K)
            seq_len_q,         // n: columns of op(Q)
            head_dim,          // k: common dimension
            &alpha,            // scaling factor
            k_reshaped.data_ptr<float>(),   // key matrix
            head_dim,          // leading dimension of K
            q_reshaped.data_ptr<float>(),   // query matrix
            head_dim,          // leading dimension of Q
            &beta,             // scaling factor for C
            attention_scores.data_ptr<float>(),  // output scores
            seq_len_k          // leading dimension of scores
        ));

    } else {
        // Handle other data types if needed
        AT_ERROR("Unsupported data type for pruned_attention_forward");
    }
    
    // Reshape attention scores
    attention_scores = attention_scores.view({batch_size, num_heads, seq_len_q, seq_len_k});
    
    // Apply mask if provided
    //if (mask.defined()) {
    //    attention_scores = attention_scores + mask;
    //}
    
    // Apply softmax - no direct cuBLAS equivalent, use PyTorch's implementation
    auto attention_probs = torch::softmax(attention_scores, -1);
    
    // Reshape back to [batch_size * num_heads, seq_len_q, seq_len_k]
    attention_probs = attention_probs.view({batch_size * num_heads, seq_len_q, seq_len_k});
    
    // Calculate context: attention_probs * V
    auto context = torch::empty({batch_size * num_heads, seq_len_q, head_dim}, query.options());
    
    if (query.dtype() == torch::kFloat) {
        float alpha = 1.0f;
        float beta = 0.0f;
        
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,       // No transpose for values
            CUBLAS_OP_N,       // No transpose for probs
            head_dim,          // m: rows of op(V)
            seq_len_q,         // n: columns of op(probs)
            seq_len_k,         // k: common dimension
            &alpha,            // scaling factor
            v_reshaped.data_ptr<float>(),   // value matrix
            head_dim,          // leading dimension of V
            attention_probs.data_ptr<float>(),  // attention probs
            seq_len_k,         // leading dimension of probs
            &beta,             // scaling factor for C
            context.data_ptr<float>(),      // output context
            head_dim           // leading dimension of context
        ));
    } else {
        // Handle other data types if needed
        AT_ERROR("Unsupported data type for pruned_attention_forward");
    }
    
    // Reshape context back to [batch_size, seq_len_q, num_heads, head_dim]
    context = context.view({batch_size, num_heads, seq_len_q, head_dim})
                    .permute({0, 2, 1, 3})
                    .contiguous();
    
    return context;
}


// Implement output projection
torch::Tensor pruned_output_projection(
    torch::Tensor context,
    torch::Tensor out_weight,
    torch::Tensor out_bias,
    const std::vector<int>& head_indices,
    int orig_num_heads,
    int head_dim) {
    
    // Ensure input is on CUDA
    TORCH_CHECK(context.device().is_cuda(), "context must be a CUDA tensor");
    TORCH_CHECK(out_weight.device().is_cuda(), "out_weight must be a CUDA tensor");
    
    // Get tensor dimensions
    int batch_size = context.size(0);
    int seq_len = context.size(1);
    int num_remaining_heads = head_indices.size();
    int embed_dim = out_weight.size(0);  // Output dimension
    
    // Debug prints
    /*
    fprintf(stderr, "Context shape: [%ld, %ld, %ld, %ld]\n", 
            context.size(0), context.size(1), context.size(2), context.size(3));
    fprintf(stderr, "Out weight shape: [%ld, %ld]\n", 
            out_weight.size(0), out_weight.size(1));
    fprintf(stderr, "Remaining heads: %d, Head dim: %d\n", 
            num_remaining_heads, head_dim);
    */
   
    // Extract pruned weights for output projection
    // out_weight shape: [embed_dim, orig_num_heads * head_dim]
    // We need to extract columns corresponding to remaining heads
    auto out_weight_per_head = out_weight.view({embed_dim, orig_num_heads, head_dim});
    auto pruned_out_weight = torch::zeros({embed_dim, num_remaining_heads, head_dim}, 
                                         out_weight.options());
    
    // Copy weights for remaining heads
    for (int i = 0; i < num_remaining_heads; i++) {
        int head_idx = head_indices[i];
        pruned_out_weight.select(1, i).copy_(out_weight_per_head.select(1, head_idx));
    }
    
    // Reshape back to 2D: [embed_dim, num_remaining_heads * head_dim]
    pruned_out_weight = pruned_out_weight.reshape({embed_dim, num_remaining_heads * head_dim});
    
    //fprintf(stderr, "Pruned out weight shape: [%ld, %ld]\n", pruned_out_weight.size(0), pruned_out_weight.size(1));
    
    // Reshape context for matrix multiplication
    // From [batch_size, seq_len, num_remaining_heads, head_dim] 
    // To [batch_size * seq_len, num_remaining_heads * head_dim]
    auto context_reshaped = context.reshape({batch_size * seq_len, num_remaining_heads * head_dim})
                                  .contiguous();
    
    //fprintf(stderr, "Context reshaped: [%ld, %ld]\n", context_reshaped.size(0), context_reshaped.size(1));
    
    // Create output tensor
    auto output = torch::empty({batch_size * seq_len, embed_dim}, context.options());
    
    // Get cuBLAS handle
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Set cuBLAS stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    
    // Perform output projection using cuBLAS
    // We want: output = context * pruned_out_weight^T
    // context: [batch*seq, num_heads*head_dim]
    // pruned_out_weight: [embed_dim, num_heads*head_dim]  
    // output: [batch*seq, embed_dim]
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    if (context.dtype() == torch::kFloat) {
        CHECK_CUBLAS(cublasSgemm_v2(
            handle,
            CUBLAS_OP_T,                      // Transpose pruned_out_weight 
            CUBLAS_OP_T,                      // Transpose context_reshaped
            embed_dim,                        // m: rows of output
            batch_size * seq_len,             // n: cols of output
            num_remaining_heads * head_dim,   // k: common dimension
            &alpha,                           // scaling factor
            pruned_out_weight.data_ptr<float>(), // A matrix (weight)
            num_remaining_heads * head_dim,   // lda: leading dimension of A
            context_reshaped.data_ptr<float>(), // B matrix (context)
            num_remaining_heads * head_dim,   // ldb: leading dimension of B
            &beta,                            // scaling factor for C
            output.data_ptr<float>(),         // C matrix (output)
            embed_dim                         // ldc: leading dimension of C
        ));
    } else {
        AT_ERROR("Unsupported data type for pruned_output_projection");
    }
    
    // Transpose output to get correct shape
    output = output.transpose(0, 1).contiguous();  // [embed_dim, batch*seq]
    
    // Add bias if defined
    if (out_bias.defined()) {
        fprintf(stderr, "Adding bias with shape: [%ld]\n", out_bias.size(0));
        output = output + out_bias.unsqueeze(1);  // Broadcast bias across sequence dimension
    }
    
    // Transpose back to final shape
    output = output.transpose(0, 1).contiguous();  // [batch*seq, embed_dim]
    
    // Reshape output back to [batch_size, seq_len, embed_dim]
    output = output.view({batch_size, seq_len, embed_dim});
    
    //fprintf(stderr, "Final output shape: [%ld, %ld, %ld]\n", output.size(0), output.size(1), output.size(2));
    
    return output;
}