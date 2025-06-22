#include "pruned_attention.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("pruned_qkv_projection_int8", &pruned_qkv_projection_int8, 
        "Optimized QKV projection for pruned attention heads",
        py::arg("hidden_states"),
        py::arg("q_weight"),
        py::arg("k_weight"),
        py::arg("v_weight"),
        py::arg("q_bias"),
        py::arg("k_bias"),
        py::arg("v_bias"),
        py::arg("head_indices"),
        py::arg("orig_num_heads"),
        py::arg("head_dim"),
        py::arg("fused") = false,
        py::arg("return_full_size") = true);

    m.def("pruned_qkv_projection", &pruned_qkv_projection, 
        "Optimized QKV projection for pruned attention heads",
        py::arg("hidden_states"),
        py::arg("q_weight"),
        py::arg("k_weight"),
        py::arg("v_weight"),
        py::arg("q_bias"),
        py::arg("k_bias"),
        py::arg("v_bias"),
        py::arg("head_indices"),
        py::arg("orig_num_heads"),
        py::arg("head_dim"),
        py::arg("fused") = false,
        py::arg("return_full_size") = true);

    m.def("pruned_attention_forward", &pruned_attention_forward,
            "Optimized attention for pruned heads",
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            //py::arg("attention_mask"),
            py::arg("scale"));

    m.def("pruned_output_projection", &pruned_output_projection,
            "Optimized output projection for pruned attention heads",
            py::arg("context"),
            py::arg("out_weight"),
            py::arg("out_bias"),
            py::arg("head_indices"),
            py::arg("orig_num_heads"),
            py::arg("head_dim"));

}