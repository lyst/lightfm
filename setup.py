# coding=utf-8
import os
import subprocess
import sys
import textwrap

from setuptools import Command, Extension, setup
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext

# Import version even when extensions are not yet built
__builtins__.__LIGHTFM_SETUP__ = True
from lightfm import __version__ as version  # NOQA


class LightfmBuildExt(build_ext):
    """
    Configures compilation options depending on compiler.
    """

    def make_compiler_args(self):
        compiler = self.compiler.compiler_type
        compile_args = []
        link_args = []

        if compiler == 'msvc':
            if use_openmp:
                compile_args.append('-openmp')

        elif compiler in ('gcc', 'unix'):
            compile_args.append('-ffast-math')

            # There are problems with illegal ASM instructions
            # when using the Anaconda distribution (at least on OSX).
            # This could be because Anaconda uses its own assembler?
            # To work around this we do not add -march=native if we
            # know we're dealing with Anaconda
            if 'anaconda' not in sys.version.lower():
                compile_args.append('-march=native')

            if use_openmp:
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')

        print('Use openmp:', use_openmp)
        print('Compiler:', compiler)
        print('Compile args:', compile_args)
        print('Link args:', link_args)

        return compile_args, link_args

    def build_extensions(self):
        compile_args, link_args = self.make_compiler_args()
        for extension in self.extensions:
            extension.extra_compile_args.extend(compile_args)
            extension.extra_link_args.extend(link_args)
        super().build_extensions()


def define_extensions(use_openmp):
    if not use_openmp:
        print('Compiling without OpenMP support.')
        return [Extension("lightfm._lightfm_fast_no_openmp",
                          ['lightfm/_lightfm_fast_no_openmp.c'])]
    else:
        return [Extension("lightfm._lightfm_fast_openmp",
                          ['lightfm/_lightfm_fast_openmp.c'])]


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
        openmp_import = textwrap.dedent("""
             from cython.parallel import parallel, prange
             cimport openmp
        """)

        lock_init = textwrap.dedent("""
             cdef openmp.omp_lock_t THREAD_LOCK
             openmp.omp_init_lock(&THREAD_LOCK)
        """)

        params = (('no_openmp', dict(openmp_import='',
                                     nogil_block='with nogil:',
                                     range_block='range',
                                     thread_num='0',
                                     lock_init='',
                                     lock_acquire='',
                                     lock_release='')),
                  ('openmp', dict(openmp_import=openmp_import,
                                  nogil_block='with nogil, parallel(num_threads=num_threads):',
                                  range_block='prange',
                                  thread_num='openmp.omp_get_thread_num()',
                                  lock_init=lock_init,
                                  lock_acquire='openmp.omp_set_lock(&THREAD_LOCK)',
                                  lock_release='openmp.omp_unset_lock(&THREAD_LOCK)')))

        file_dir = os.path.join(os.path.dirname(__file__),
                                'lightfm')

        with open(os.path.join(file_dir,
                               '_lightfm_fast.pyx.template'), 'r') as fl:
            template = fl.read()

        for variant, template_params in params:
            with open(os.path.join(file_dir,
                                   '_lightfm_fast_{}.pyx'.format(variant)),
                      'w') as fl:
                fl.write(template.format(**template_params))

    def run(self):
        from Cython.Build import cythonize

        self.generate_pyx()

        cythonize([Extension("lightfm._lightfm_fast_no_openmp",
                             ['lightfm/_lightfm_fast_no_openmp.pyx']),
                   Extension("lightfm._lightfm_fast_openmp",
                             ['lightfm/_lightfm_fast_openmp.pyx'])])


class Clean(Command):
    """
    Clean build files.
    """

    user_options = [
        ('all', None, '(Compatibility with original clean command)')
    ]

    def initialize_options(self):
        self.all = False

    def finalize_options(self):
        pass

    def run(self):
        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, 'lightfm.egg-info')])
        subprocess.call(
            ['find', pth, '-name', 'lightfm*.pyc', '-type', 'f', '-delete'])
        subprocess.call(
            ['rm', os.path.join(pth, 'lightfm', '_lightfm_fast.so')])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests/']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


use_openmp = not sys.platform.startswith('darwin')

setup(
    name='lightfm',
    version=version,
    description='LightFM recommendation model',
    url='https://github.com/lyst/lightfm',
    download_url='https://github.com/lyst/lightfm/tarball/{}'.format(version),
    packages=['lightfm',
              'lightfm.datasets'],
    package_data={'': ['*.c']},
    install_requires=['numpy', 'scipy>=0.17.0', 'requests', 'scikit-learn'],
    tests_require=['pytest', 'requests', 'scikit-learn'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean, 'build_ext': LightfmBuildExt},
    author='Lyst Ltd (Maciej Kula)',
    author_email='data@ly.st',
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    ext_modules=define_extensions(use_openmp)
)
