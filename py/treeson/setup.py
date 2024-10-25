from setuptools import setup
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        'treeson',
        ['src/bindings.cpp']),
]

setup(
    name='treeson',
    version='0.0.1',
    author='JS',
    author_email='szitas.juraj13@example.com',
    description='Pybind11 bindings for treeson',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)

