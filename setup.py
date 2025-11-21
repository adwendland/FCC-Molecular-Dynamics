from setuptools import setup, Extension
import sys
import pybind11

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-std=c++17"]

ext_modules = [
    Extension(
        "md.md_cpp",              # package.module name
        ["md/md_cpp.cpp"],        # 
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="md_cpp",
    version="0.1",
    ext_modules=ext_modules,
)
