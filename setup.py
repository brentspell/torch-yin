#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(
    name="torch-yin",
    version="0.1.2",
    description="Yin pitch estimator for PyTorch",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/brentspell/torch-yin/",
    author="Brent M. Spell",
    author_email="brent@brentspell.com",
    packages=setuptools.find_packages(),
    setup_requires=[],
    install_requires=["numpy>=1.20", "torch>=1.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
