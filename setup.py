import os
import platform
import subprocess
import sys

from setuptools import setup, Command, Extension
from setuptools.command.test import test as TestCommand


# Use gcc for openMP on OSX
if 'darwin' in platform.platform().lower():
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "g++-4.9"


def define_extensions(file_ext):

    return [Extension("lightfm.lightfm_fast",
                      ['lightfm/lightfm_fast%s' % file_ext],
                      extra_link_args=["-fopenmp"],
                      extra_compile_args=['-fopenmp'])]


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        import Cython
        from Cython.Build import cythonize

        cythonize(define_extensions('.pyx'))


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
        subprocess.call(['rm', os.path.join(pth, 'lightfm', 'lightfm_fast.so')])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='lightfm',
    version='0.0.1',
    description='LightFM recommendation model',
    url='https://github.com/lyst/lightfm',
    download_url='https://github.com/lyst/lightfm/tarball/1.0',
    packages=['lightfm'],
    install_requires=['numpy'],
    tests_require=['pytest', 'requests', 'scikit-learn', 'scipy'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    author='Lyst Ltd (Maciej Kula)',
    author_email='data@ly.st',
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    ext_modules=define_extensions('.c')
)
