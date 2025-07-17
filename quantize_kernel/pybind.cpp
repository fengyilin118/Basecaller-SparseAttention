#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm_cuda.h"



PYBIND11_MODULE(awq_inference_engine, m)
{

    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new, "New quantized GEMM kernel.");
 
}