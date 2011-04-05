#!/usr/bin/env python
# -*- coding: latin1 -*-

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py

setup(name="compyte",
      version="2011.1",
      description="A common set of compute primitives for PyCUDA and PyOpenCL (to be created)",
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        ],

      install_requires=[
          "decorator>=3.2.0"
          ],

      author="Andreas Kloeckner",
      url="http://pypi.python.org/pypi/compyte",
      author_email="inform@tiker.net",
      license = "MIT",
      packages=["compyte"],

      # 2to3 invocation
      cmdclass={'build_py': build_py})
