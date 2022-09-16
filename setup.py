# coding=utf-8
import os
import pathlib
import subprocess
import sys
import textwrap

from setuptools import Command, Extension, setup


def define_extensions(use_openmp):
    compile_args = []
    if not os.environ.get("LIGHTFM_NO_CFLAGS"):
        compile_args += ["-ffast-math"]

        if sys.platform.startswith("darwin"):
            compile_args += []
        else:
            compile_args += ["-march=native"]

    if not use_openmp:
        print("Compiling without OpenMP support.")
        return [
            Extension(
                "lightfm._lightfm_fast_no_openmp",
                ["lightfm/_lightfm_fast_no_openmp.c"],
                extra_compile_args=compile_args,
            )
        ]
    else:
        return [
            Extension(
                "lightfm._lightfm_fast_openmp",
                ["lightfm/_lightfm_fast_openmp.c"],
                extra_link_args=["-fopenmp"],
                extra_compile_args=compile_args + ["-fopenmp"],
            )
        ]


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def generate_pyx(self):
        openmp_import = textwrap.dedent(
            """
             from cython.parallel import parallel, prange
             cimport openmp
        """
        )

        lock_init = textwrap.dedent(
            """
             cdef openmp.omp_lock_t THREAD_LOCK
             openmp.omp_init_lock(&THREAD_LOCK)
        """
        )

        params = (
            (
                "no_openmp",
                dict(
                    openmp_import="",
                    nogil_block="with nogil:",
                    range_block="range",
                    thread_num="0",
                    lock_init="",
                    lock_acquire="",
                    lock_release="",
                ),
            ),
            (
                "openmp",
                dict(
                    openmp_import=openmp_import,
                    nogil_block="with nogil, parallel(num_threads=num_threads):",
                    range_block="prange",
                    thread_num="openmp.omp_get_thread_num()",
                    lock_init=lock_init,
                    lock_acquire="openmp.omp_set_lock(&THREAD_LOCK)",
                    lock_release="openmp.omp_unset_lock(&THREAD_LOCK)",
                ),
            ),
        )

        file_dir = os.path.join(os.path.dirname(__file__), "lightfm")

        with open(os.path.join(file_dir, "_lightfm_fast.pyx.template"), "r") as fl:
            template = fl.read()

        for variant, template_params in params:
            with open(
                os.path.join(file_dir, "_lightfm_fast_{}.pyx".format(variant)), "w"
            ) as fl:
                fl.write(template.format(**template_params))

    def run(self):
        from Cython.Build import cythonize

        self.generate_pyx()

        cythonize(
            [
                Extension(
                    "lightfm._lightfm_fast_no_openmp",
                    ["lightfm/_lightfm_fast_no_openmp.pyx"],
                ),
                Extension(
                    "lightfm._lightfm_fast_openmp",
                    ["lightfm/_lightfm_fast_openmp.pyx"],
                    extra_link_args=["-fopenmp"],
                ),
            ]
        )


class Clean(Command):
    """
    Clean build files.
    """

    user_options = [("all", None, "(Compatibility with original clean command)")]

    def initialize_options(self):
        self.all = False

    def finalize_options(self):
        pass

    def run(self):
        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(["rm", "-rf", os.path.join(pth, "build")])
        subprocess.call(["rm", "-rf", os.path.join(pth, "lightfm.egg-info")])
        subprocess.call(["find", pth, "-name", "lightfm*.pyc", "-type", "f", "-delete"])
        subprocess.call(["rm", os.path.join(pth, "lightfm", "_lightfm_fast.so")])


def read_version():
    mod = {}
    path = os.path.join(
        os.path.dirname(__file__),
        "lightfm",
        "version.py",
    )
    with open(path) as fd:
        exec(fd.read(), mod)
    return mod["__version__"]


use_openmp = not sys.platform.startswith("darwin") and not sys.platform.startswith(
    "win"
)

long_description = pathlib.Path(__file__).parent.joinpath("README.md").read_text()

setup(
    name="lightfm",
    version=read_version(),
    description="LightFM recommendation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyst/lightfm",
    download_url="https://github.com/lyst/lightfm/tarball/{}".format(read_version()),
    packages=["lightfm", "lightfm.datasets"],
    package_data={"": ["*.c"]},
    install_requires=["numpy", "scipy>=0.17.0", "requests", "scikit-learn"],
    tests_require=["pytest", "requests", "scikit-learn"],
    cmdclass={"cythonize": Cythonize, "clean": Clean},
    author="Lyst Ltd (Maciej Kula)",
    author_email="data@ly.st",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=define_extensions(use_openmp),
)
