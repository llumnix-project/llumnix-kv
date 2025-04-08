from setuptools import setup

from setup_common import CustomBuildExtension, blade_kvt_ext

# rm -rf blade_llm.egg blade_kvt.egg dist/
setup(
    name="blade_kvt",
    version="1.0.0",
    packages=["blade_kvt"],
    author="Alibaba PAI Team",
    ext_modules=[blade_kvt_ext],
    cmdclass={
        "build_ext": CustomBuildExtension,
    },
)
