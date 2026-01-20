# coding=utf-8
"""
Legacy setup script for building LightFM.

Note: The primary build system is now maturin (configured in pyproject.toml).
This file is retained for backwards compatibility with older build workflows.

To build with the new Rust backend:
    pip install maturin
    maturin develop

To build a wheel:
    maturin build --release
"""
import os
import subprocess
import sys

from setuptools import Command, setup


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
        subprocess.call(["rm", "-rf", os.path.join(pth, "target")])
        subprocess.call(["rm", "-rf", os.path.join(pth, "lightfm.egg-info")])
        subprocess.call(["find", pth, "-name", "lightfm*.pyc", "-type", "f", "-delete"])
        subprocess.call(["find", pth, "-name", "*.so", "-type", "f", "-delete"])


setup(
    cmdclass={"clean": Clean},
)
