import os
import subprocess
import sys
import textwrap

from setuptools import Command, Extension, setup
from setuptools.command.test import test as TestCommand


def define_extensions(use_openmp):

    compile_args = ['-ffast-math']

    # There are problems with illegal ASM instructions
    # when using the Anaconda distribution (at least on OSX).
    # This could be because Anaconda uses its own assembler?
    # To work around this we do not add -march=native if we
    # know we're dealing with Anaconda
    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')

    if not use_openmp:
        print('Compiling without OpenMP support.')
        return [Extension("lightfm._lightfm_fast_no_openmp",
                          ['lightfm/_lightfm_fast_no_openmp.c'],
                          extra_compile_args=compile_args)]
    else:
        return [Extension("lightfm._lightfm_fast_openmp",
                          ['lightfm/_lightfm_fast_openmp.c'],
                          extra_link_args=["-fopenmp"],
                          extra_compile_args=compile_args + ['-fopenmp'])]


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

        params = (('no_openmp', dict(openmp_import='',
                                     nogil_block='with nogil:',
                                     range_block='range',
                                     thread_num='0')),
                  ('openmp', dict(openmp_import=openmp_import,
                                  nogil_block='with nogil, parallel(num_threads=num_threads):',
                                  range_block='prange',
                                  thread_num='openmp.omp_get_thread_num()')))

        file_dir = os.path.join(os.path.dirname(__file__),
                                'lightfm')

        with open(os.path.join(file_dir,
                               '_lightfm_fast.pyx.template'), 'r') as fl:
            template = fl.read()

        for variant, template_params in params:
            with open(os.path.join(file_dir,
                                   '_lightfm_fast_{}.pyx'.format(variant)), 'w') as fl:
                fl.write(template.format(**template_params))

    def run(self):

        from Cython.Build import cythonize

        self.generate_pyx()

        cythonize([Extension("lightfm._lightfm_fast_no_openmp",
                             ['lightfm/_lightfm_fast_no_openmp.pyx']),
                   Extension("lightfm._lightfm_fast_openmp",
                             ['lightfm/_lightfm_fast_openmp.pyx'],
                             extra_link_args=['-fopenmp'])])


class Clean(Command):
    """
    Clean build files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, 'lightfm.egg-info')])
        subprocess.call(['find', pth, '-name', 'lightfm*.pyc', '-type', 'f', '-delete'])
        subprocess.call(['rm', os.path.join(pth, 'lightfm', '_lightfm_fast.so')])


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


use_openmp = not (sys.platform.startswith('darwin') or sys.platform.startswith('win'))


setup(
    name='lightfm',
    version='1.11',
    description='LightFM recommendation model',
    url='https://github.com/lyst/lightfm',
    download_url='https://github.com/lyst/lightfm/tarball/1.11',
    packages=['lightfm',
              'lightfm.datasets'],
    package_data={'': ['*.c']},
    install_requires=['numpy', 'scipy>=0.17.0', 'requests'],
    tests_require=['pytest', 'requests', 'scikit-learn'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    author='Lyst Ltd (Maciej Kula)',
    author_email='data@ly.st',
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    ext_modules=define_extensions(use_openmp)
)
