#!/usr/bin/env python3
# Builds the `uds_loader` C++ extension from the FLAT sft_final/ layout.
#   python setup.py build_ext --inplace
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
setup(
    name="uds_loader",
    version="0.1.0",
    ext_modules=[CppExtension(
        name="uds_loader",
        sources=[os.path.join(HERE, "bindings.cpp")],
        include_dirs=[HERE],
        extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden"],
        extra_link_args=["-lpthread"],
    )],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
