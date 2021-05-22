from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cu_gemm_2x48',
    ext_modules=[
        CUDAExtension('cu_gemm_2x48', [
            'gemm_mt.cpp',
            'gemm_mt_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
