#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <chrono>


// Forward declarations
struct TensorCoreSpecs;
class RTX3060Optimizer;
class TensorCoreAligner;
class AdaptiveTensorCoreAligner;


struct TensorCoreSpecs {
    int fp16_multiple;
    int bf16_multiple; 
    int int8_multiple;
    int preferred_tile_size;
    int compute_capability_major;
    int compute_capability_minor;
    std::string architecture_name;
};

extern const std::map<std::string, TensorCoreSpecs> GPU_SPECS;

class RTX3060Optimizer {
private:
    TensorCoreSpecs specs;
    static constexpr int NUM_SMS = 28;              ///< RTX 3060 has 28 SMs
    static constexpr size_t TOTAL_VRAM = 12ULL * 1024 * 1024 * 1024;  ///< 12GB VRAM
    static constexpr size_t L1_CACHE_SIZE = 192 * 1024;              ///< 192KB L1 per SM
    static constexpr size_t L2_CACHE_SIZE = 3 * 1024 * 1024;         ///< 3MB L2 total
    
public:

    RTX3060Optimizer();

    int get_optimal_batch_size(int target_batch_size);

    std::tuple<int, int, int> get_memory_efficient_dims(
        int m, int n, int k, 
        torch::Dtype dtype,
        float memory_utilization=0.8f);

    ~RTX3060Optimizer() = default;
};