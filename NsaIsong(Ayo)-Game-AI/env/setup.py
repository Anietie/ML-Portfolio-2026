import sys
from setuptools import setup, Extension
import pybind11

# Platform-specific compiler flags
if sys.platform == "win32":
    compile_args = ['/O2', '/std:c++17']
else:
    compile_args = ['-O3', '-std=c++17']

ext_modules = [
    Extension(
        'cpp_env',
        ['cpp_env.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args
    ),
]

setup(
    name='cpp_env',
    ext_modules=ext_modules,
)