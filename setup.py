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
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

int main(int argc, char** argv)
{
    // 固定要查找的符号
    const char* symbol = "cudaMemcpyBatchAsync";

    // 自动查找 libcudart.so 的路径列表
    char lib_paths[10][512];
    int path_count = 0;

    // 标准路径
    strcpy(lib_paths[path_count++], "/usr/local/cuda/lib64/libcudart.so");
    strcpy(lib_paths[path_count++], "/usr/local/cuda/lib/libcudart.so");
    strcpy(lib_paths[path_count++], "/usr/lib/x86_64-linux-gnu/libcudart.so");
    strcpy(lib_paths[path_count++], "/usr/lib/libcudart.so");

    // 检查环境变量中的 CUDA 路径
    const char* cuda_home = getenv("CUDA_HOME");
    const char* cuda_path = getenv("CUDA_PATH");

    if (cuda_home) {
        snprintf(lib_paths[path_count], sizeof(lib_paths[0]), "%s/lib64/libcudart.so", cuda_home);
        path_count++;
        snprintf(lib_paths[path_count], sizeof(lib_paths[0]), "%s/lib/libcudart.so", cuda_home);
        path_count++;
    }
    if (cuda_path) {
        snprintf(lib_paths[path_count], sizeof(lib_paths[0]), "%s/lib64/libcudart.so", cuda_path);
        path_count++;
        snprintf(lib_paths[path_count], sizeof(lib_paths[0]), "%s/lib/libcudart.so", cuda_path);
        path_count++;
    }

    for (int i = 0; i < path_count; i++) {
        const char* lib_path = lib_paths[i];
        void* handle = dlopen(lib_path, RTLD_LAZY);
        if (handle) {
            dlerror();
            void* sym = dlsym(handle, symbol);
            const char* err = dlerror();

            if (err) {
                printf("Symbol '%s' NOT found in %s\\n", symbol, lib_path);
            } else {
                printf("FOUND\\n");
                dlclose(handle);
                return 0;
            }

            dlclose(handle);
        }
    }
    printf("NOT_FOUND\\n");
    return 1;
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
            test_binary = os.path.join(tmpdir, "test_cuda_batch")
            compile_result = subprocess.run(
                ["gcc", "-o", test_binary, test_file, "-ldl"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                print(f"Failed to compile test program: {compile_result.stderr}")
                return False

            # Run the test program (it will automatically find libcudart.so and check symbol)
            run_result = subprocess.run(
                [test_binary],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check output: "FOUND" means symbol exists, "NOT_FOUND" means it doesn't
            if run_result.returncode == 0 and "FOUND" in run_result.stdout:
                print("CUDA batch copy (cudaMemcpyBatchAsync) is supported")
                return True
            else:
                print("CUDA batch copy (cudaMemcpyBatchAsync) is not supported")
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
# If PPU_VERSION_NUM environment variable exists, force disable batch copy
if "PPU_VERSION_NUM" in os.environ:
    _build_options.append("-DENABLE_BATCH_COPY=OFF")
    print(f"PPU_VERSION_NUM={os.environ['PPU_VERSION_NUM']} detected, forcing -DENABLE_BATCH_COPY=OFF")
elif check_cuda_batch_copy_support():
    _build_options.append("-DENABLE_BATCH_COPY=ON")
    print("Adding -DENABLE_BATCH_COPY=ON to build options")
else:
    _build_options.append("-DENABLE_BATCH_COPY=OFF")
    print("Adding -DENABLE_BATCH_COPY=OFF to build options")

blade_kvt_ext = CMakeExtension("blade_kvt.kvtransfer_ops", src_dir=_kvtransfer_src, build_options=_build_options)


def _barex_ver():
    if "PPU_VERSION_NUM" in os.environ:
        return "unknown.ppu"
    return "unknown.cuda" # 等待 xiaoshi 修复 barex benchmark
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
