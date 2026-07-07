#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py — builds the `uds_loader` C++ extension against the installed libtorch.

    cd python && pip install -e .        # or: python setup.py build_ext --inplace

Requires: torch (with matching CUDA), a C++17 compiler. The extension links the
libtorch that ships with your installed `torch` wheel, so CUDA/toolchain match
automatically.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
# realpath() collapses the ".." so distutils' temp build tree stays valid
INCLUDE = os.path.realpath(os.path.join(HERE, "..", "cpp", "include"))
SRC = os.path.realpath(os.path.join(HERE, "..", "cpp", "src", "bindings.cpp"))

setup(
    name="uds_loader",
    version="0.1.0",
    description="UDS SFT C++ data pipeline (reader/queues/scheduler/collator/prefetch/DDP)",
    ext_modules=[
        CppExtension(
            name="uds_loader",
            sources=[SRC],
            include_dirs=[INCLUDE],
            extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden"],
            extra_link_args=["-lpthread"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
