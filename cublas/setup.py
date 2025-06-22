from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pruned_attention_cublas',
    ext_modules=[
        CUDAExtension('pruned_attention_cublas', [
            'pruned_attention.cpp',
            'pruned_attention_cuda.cu',
            'tensor_core_optimizer.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)