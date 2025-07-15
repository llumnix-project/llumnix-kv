from setuptools import setup, Extension
from setuptools_scm import get_version
from setuptools_scm.version import get_local_node_and_date

import os
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
blade_kvt_ext = CMakeExtension("blade_kvt.kvtransfer_ops", src_dir=_kvtransfer_src, build_options=_build_options)


def _local_version(version) -> str:
    local_ver = get_local_node_and_date(version)
    version_parts = [local_ver]
    if int(os.environ.get("BLADELLM_CMAKE_DEBUG", 0)):
        version_parts.append('debug')
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
