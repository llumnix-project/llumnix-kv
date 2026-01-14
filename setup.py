from setuptools import setup, Extension
from setuptools_scm import get_version
from setuptools_scm.version import get_local_node_and_date

import os
import re
import subprocess
from pathlib import Path
from shutil import which
from typing import List, Union

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
)


def is_ninja_available() -> bool:
    return which("ninja") is not None


def check_cuda_batch_copy_support() -> bool:
    """
    Check if CUDA supports cudaMemcpyBatchAsync by compiling a test program.
    Returns True if cudaMemcpySrcAccessOrderStream enum is defined (CUDA 12.8+).
    """
    import tempfile
    
    # Test program to check if cudaMemcpySrcAccessOrderStream enum is available
    # This enum is available in CUDA 12.8+ which also provides cudaMemcpyBatchAsync
    test_code = """
#include <cuda_runtime.h>

int main() {
#if defined(cudaMemcpySrcAccessOrderStream)
    // Enum is defined, feature is available
    return 0;
#else
    // Enum is not defined, feature is not available
    return 1;
#endif
}
"""
    
    try:
        # Check if nvcc is available
        if not which("nvcc"):
            print("nvcc not found, assuming CUDA batch copy is not supported")
            return False
        
        # Create temporary directory for test compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_cuda_batch.cpp")
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            # Try to compile the test program
            result = subprocess.run(
                ["nvcc", "-o", os.path.join(tmpdir, "test_cuda_batch"), test_file, "-lcudart"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("CUDA batch copy (cudaMemcpyBatchAsync) is supported")
                return True
            else:
                print(f"CUDA batch copy is not supported (CUDA < 12.8)")
                return False
                
    except Exception as e:
        print(f"Error checking CUDA batch copy support: {e}, assuming not supported")
        return False


class CMakeExtension(Extension):
    def __init__(self, name: str, src_dir: str = ".", build_options: Union[str, List[str]] = None) -> None:
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(src_dir)
        if build_options:
            if isinstance(build_options, str):
                self.build_options = [build_options]
            else:
                self.build_options = build_options


class CustomBuildExtension(BuildExtension):
    def build_cmake_extension(self, ext: CMakeExtension):
        ext_output = self.get_ext_filename(ext.name)
        ext_suffix = ext_output.split('.')[-2]
        debug = int(os.environ.get("BLADELLM_CMAKE_DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        torch_dir = torch.utils.cmake_prefix_path
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DTorch_DIR={torch_dir}/Torch",
            f"-DTARGET_SUFFIX={ext_suffix}",
        ]
        cmake_args += ext.build_options
        build_tools = []
        if is_ninja_available():
            build_tools = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(4),
            ]

        cmdline = ["cmake", ext.source_dir, *build_tools, *cmake_args]
        print(f"start to build cmake extension: {ext.name} as {ext_output} use {cmdline} ...")
        subprocess.run(cmdline, cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", "-j=4"], cwd=build_temp, check=True)

    def build_extension(self, ext: Extension) -> None:
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)


_kvtransfer_src = os.path.join(os.getcwd(), "kvtransfer")
_build_options = ["-DBUILD_TESTS=OFF", "-DBUILD_RDMA=ON", "-DBUILD_PYTHON_BIND=ON"]

# Check CUDA batch copy support and add corresponding CMake option
if check_cuda_batch_copy_support():
    _build_options.append("-DENABLE_BATCH_COPY=ON")
    print("Adding -DENABLE_BATCH_COPY=ON to build options")
else:
    _build_options.append("-DENABLE_BATCH_COPY=OFF")
    print("Adding -DENABLE_BATCH_COPY=OFF to build options")

blade_kvt_ext = CMakeExtension("blade_kvt.kvtransfer_ops", src_dir=_kvtransfer_src, build_options=_build_options)


def _barex_ver():
    return "unknown"  # 等待 xiaoshi 修复 barex benchmark
    result = subprocess.run(
        ["barex_benchmark", "-V"],
        capture_output=True,
        text=True,
        check=True
    )
    for line in result.stdout.splitlines():
        if "Git Message" not in line:
            continue
        # line format: printf("Git Message: %s, %s, %s\n", BUILD_BRANCH, BUILD_COMMIT_ID, BUILD_COMMIT);
        parts = line.split(',')
        return parts[1].strip()
    raise RuntimeError(f"barex_ver: failed. {result=}")

def _local_version(version) -> str:
    local_ver = get_local_node_and_date(version)
    version_parts = [local_ver]
    if int(os.environ.get("BLADELLM_CMAKE_DEBUG", 0)):
        version_parts.append('debug')
    version_parts.append(f"barex.{_barex_ver()}")
    return '.'.join(version_parts)


def get_kvt_version() -> str:
    git_describe_command = [
        "git", "describe", "--dirty", "--tags", "--long", "--match",
        "v*[0-9]*[0-9]*[0-9]"
    ]
    version = get_version(write_to="_version.py",
                          local_scheme=_local_version,
                          git_describe_command=git_describe_command)
    return version

# rm -rf blade_llm.egg blade_kvt.egg dist/
setup(
    name="blade_kvt",
    version=get_kvt_version(),
    packages=["blade_kvt"],
    author="Alibaba PAI Team",
    ext_modules=[blade_kvt_ext],
    cmdclass={
        "build_ext": CustomBuildExtension,
    },
)
