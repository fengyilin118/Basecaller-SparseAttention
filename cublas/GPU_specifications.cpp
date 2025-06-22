#include <map>
#include <string>
#include <iostream>// This file defines the specifications for various NVIDIA GPUs with Tensor Cores.
#include <cuda_runtime.h>
#include <vector>
//#include <torch/torch.h>

struct TensorCoreSpecs {
    int fp16_multiple;
    int bf16_multiple; 
    int int8_multiple;
    int preferred_tile_size;
    int compute_capability_major;
    int compute_capability_minor;
    std::string architecture_name;
};

const std::map<std::string, TensorCoreSpecs> GPU_SPECS = {
    // Volta Architecture (1st Gen Tensor Cores)
    {"V100", {
        .fp16_multiple = 8,
        .bf16_multiple = 0,        // Not supported
        .int8_multiple = 16,
        .preferred_tile_size = 256,
        .compute_capability_major = 7,
        .compute_capability_minor = 0,
        .architecture_name = "Volta"
    }},
    
    // Turing Architecture (2nd Gen Tensor Cores)
    {"T4", {
        .fp16_multiple = 8,
        .bf16_multiple = 0,        // Not supported
        .int8_multiple = 16,
        .preferred_tile_size = 256,
        .compute_capability_major = 7,
        .compute_capability_minor = 5,
        .architecture_name = "Turing"
    }},
    
    {"RTX2080Ti", {
        .fp16_multiple = 8,
        .bf16_multiple = 0,        // Not supported
        .int8_multiple = 16,
        .preferred_tile_size = 256,
        .compute_capability_major = 7,
        .compute_capability_minor = 5,
        .architecture_name = "Turing"
    }},
    
    // Ampere Architecture (3rd Gen Tensor Cores)
    {"RTX3060", {
        .fp16_multiple = 16,       // 3rd gen Tensor Cores requirement
        .bf16_multiple = 16,       // Ampere supports BF16
        .int8_multiple = 32,       // Enhanced INT8 support
        .preferred_tile_size = 384, // Optimized for GA106 (RTX 3060)
        .compute_capability_major = 8,
        .compute_capability_minor = 6,
        .architecture_name = "Ampere"
    }},
    
    {"RTX3070", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512, // More compute units than 3060
        .compute_capability_major = 8,
        .compute_capability_minor = 6,
        .architecture_name = "Ampere"
    }},
    
    {"RTX3080", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512,
        .compute_capability_major = 8,
        .compute_capability_minor = 6,
        .architecture_name = "Ampere"
    }},
    
    {"RTX3090", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512,
        .compute_capability_major = 8,
        .compute_capability_minor = 6,
        .architecture_name = "Ampere"
    }},
    
    {"A100", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512,
        .compute_capability_major = 8,
        .compute_capability_minor = 0,
        .architecture_name = "Ampere"
    }},
    
    // Ada Lovelace Architecture (4th Gen Tensor Cores)
    {"RTX4060", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 384,
        .compute_capability_major = 8,
        .compute_capability_minor = 9,
        .architecture_name = "Ada Lovelace"
    }},
    
    {"RTX4090", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512,
        .compute_capability_major = 8,
        .compute_capability_minor = 9,
        .architecture_name = "Ada Lovelace"
    }},
    
    // Hopper Architecture (4th Gen Tensor Cores + FP8)
    {"H100", {
        .fp16_multiple = 16,
        .bf16_multiple = 16,
        .int8_multiple = 32,
        .preferred_tile_size = 512,
        .compute_capability_major = 9,
        .compute_capability_minor = 0,
        .architecture_name = "Hopper"
    }}
};


/*
class RTX3060Optimizer {
private:
    TensorCoreSpecs specs;
    
public:
    RTX3060Optimizer() {
       
        specs = GPU_SPECS.at("RTX3060");
        printf("Initializing RTX 3060 optimizer:\n");
        printf("  FP16 alignment: %d elements\n", specs.fp16_multiple);
        printf("  BF16 alignment: %d elements\n", specs.bf16_multiple);
        printf("  INT8 alignment: %d elements\n", specs.int8_multiple);
        printf("  Preferred tile size: %d\n", specs.preferred_tile_size);
    }
    
    // RTX 3060 has 28 SMs, optimize batch sizes accordingly
    int get_optimal_batch_size(int target_batch_size) {
        // RTX 3060 works best with batch sizes that are multiples of 28 (number of SMs)
        const int NUM_SMS = 28;
        const std::vector<int> PREFERRED_MULTIPLES = {1, 2, 4, 7, 14, 28};
        
        for (int mult : PREFERRED_MULTIPLES) {
            if (target_batch_size <= NUM_SMS * mult) {
                return NUM_SMS * mult;
            }
        }
        
        return ((target_batch_size + NUM_SMS - 1) / NUM_SMS) * NUM_SMS;
    }
    
    // Memory-aware tiling for RTX 3060 (8GB/12GB VRAM variants)
    std::tuple<int, int, int> get_memory_efficient_dims(
        int m, int n, int k, 
        torch::Dtype dtype,
        float memory_utilization = 0.8f) {
        
        // Assume 8GB variant (conservative)
        const size_t TOTAL_VRAM = 8ULL * 1024 * 1024 * 1024; // 8GB
        const size_t AVAILABLE_VRAM = TOTAL_VRAM * memory_utilization;
        
        size_t element_size = (dtype == torch::kFloat16) ? 2 : 4;
        
        // Basic alignment first
        int m_aligned = ((m + specs.fp16_multiple - 1) / specs.fp16_multiple) * specs.fp16_multiple;
        int n_aligned = ((n + specs.fp16_multiple - 1) / specs.fp16_multiple) * specs.fp16_multiple;
        int k_aligned = ((k + specs.fp16_multiple - 1) / specs.fp16_multiple) * specs.fp16_multiple;
        
        // Check memory requirements
        size_t memory_needed = (m_aligned * k_aligned + k_aligned * n_aligned + m_aligned * n_aligned) * element_size;
        
        // If too much memory, use tiling
        if (memory_needed > AVAILABLE_VRAM) {
            int tile_size = specs.preferred_tile_size;
            
            while (memory_needed > AVAILABLE_VRAM && tile_size >= specs.fp16_multiple) {
                m_aligned = std::min(m_aligned, tile_size);
                n_aligned = std::min(n_aligned, tile_size);
                
                memory_needed = (m_aligned * k_aligned + k_aligned * n_aligned + m_aligned * n_aligned) * element_size;
                tile_size -= specs.fp16_multiple;
            }
            
            printf("RTX 3060 memory-constrained tiling: [%d, %d, %d], Memory: %.1fGB\n", 
                   m_aligned, n_aligned, k_aligned, memory_needed / (1024.0 * 1024.0 * 1024.0));
        }
        
        return {m_aligned, n_aligned, k_aligned};
    }
};*/

struct RTX3060MemoryHierarchy {
    // L1 Cache: 192KB per SM (shared between threads in the SM)
    static constexpr size_t L1_CACHE_SIZE = 192 * 1024;
    
    // L2 Cache: 3MB shared across all SMs
    static constexpr size_t L2_CACHE_SIZE = 3 * 1024 * 1024;
    
    // VRAM: 8GB GDDR6
    static constexpr size_t VRAM_SIZE = 8ULL * 1024 * 1024 * 1024;
    
    // Access speeds (relative)
    static constexpr int L1_SPEED = 1;      // Fastest
    static constexpr int L2_SPEED = 4;      // 4x slower than L1
    static constexpr int VRAM_SPEED = 100;  // 100x slower than L1
    
    static void print_hierarchy() {
        printf("RTX 3060 Memory Hierarchy:\n");
        printf("L1 Cache:  %dKB per SM    - %dx speed\n", 
               (int)(L1_CACHE_SIZE/1024), L1_SPEED);
        printf("L2 Cache:  %dMB shared    - %dx speed\n", 
               (int)(L2_CACHE_SIZE/(1024*1024)), L2_SPEED);
        printf("VRAM:      %dGB           - %dx speed\n", 
               (int)(VRAM_SIZE/(1024*1024*1024)), VRAM_SPEED);
    }
};

int main() {
            printf("Cache Utilization Comparison:\n\n");
        
        // Large matrix multiplication: 2048x2048 x 2048x2048
        int large_size = 2048;
        size_t large_memory = large_size * large_size * 2; // FP16
        
        printf("WITHOUT TILING (Large matrices):\n");
        printf("Matrix size: %dx%d (%.1fMB each)\n", 
               large_size, large_size, large_memory / (1024.0 * 1024.0));
        printf("L1 Cache (192KB): Can hold %.3f%% of one matrix\n", 
               (RTX3060MemoryHierarchy::L1_CACHE_SIZE * 100.0) / large_memory);
        printf("L2 Cache (3MB): Can hold %.1f%% of one matrix\n", 
               (RTX3060MemoryHierarchy::L2_CACHE_SIZE * 100.0) / large_memory);
        printf("Result: Frequent VRAM access → 100x slower\n\n");
        
        // Tiled approach: 256x256 tiles
        int tile_size = 256;
        size_t tile_memory = tile_size * tile_size * 2; // FP16
        
        printf("WITH TILING (Small tiles):\n");
        printf("Tile size: %dx%d (%.1fKB each)\n", 
               tile_size, tile_size, tile_memory / 1024.0);
        printf("L1 Cache (192KB): Can hold %.1f tiles\n", 
               RTX3060MemoryHierarchy::L1_CACHE_SIZE / (double)tile_memory);
        printf("L2 Cache (3MB): Can hold %.1f tiles\n", 
               RTX3060MemoryHierarchy::L2_CACHE_SIZE / (double)tile_memory);
        printf("Result: Data stays in cache → 1-4x speed\n");

}