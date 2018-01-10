Installation
============

The library is routinely tested on linux and, less frequently, on
Windows and Mac OS X.  The OS most frequently tested are:

 - Debian 6
 - Ubuntu 16.04
 - macOS 10.12
 - Windows 7

It should also work on any decently recent OS not listed here. If you
get an error during the build on your favorite OS, please report it
and we will attempt to fix it.

Conda
-----

The easiest way to install libgpuarray is with conda::

  conda install pygpu

This will also install the libgpuarray package automatically.

This should work on Linux, Mac OS and Windows.

This is also available in packages in conda-forge. They could be more
up to date::

  conda install -c conda-forge pygpu

Build Requirements
------------------

 - cmake >= 3.0 (cmake_).
 - a c99-compliant compiler (or MSVC if on windows).
 - (optional) libcheck (check_) to run the C tests.
 - (optional) python (python_) for the python bindings.
 - (optional) mako (mako_) for development or running the python bindings.
 - (optional) Cython >= 0.25 (cython_) for the python bindings.
 - (optional) nosetests (nosetests_) to run the python tests.

Run Requirements
----------------

No matter what was available at build time, this library comes with
dynamic loaders for the following libraries.  You don't need to have
any of this available, but you won't be able to use associated
functionality.

 * For CUDA:

   - CUDA (cuda_) version 7.0 or more, with the appropriate driver
   - (optional) NCCL (nccl_) for the collectives interface

 * For OpenCL:

   - OpenCL version 1.2 or more
   - (optional) clBLAS (clblas_) or CLBlast (clblast_) for blas functionality

   .. note::

      The OpenCL that comes with OS X is fundamentally broken and
      doesn't work with some of the kernels in the library.  You can
      use it at your own risk, but don't report problems with it we
      can't fix them.


Download
--------

::

  git clone https://github.com/Theano/libgpuarray.git
  cd libgpuarray

Step-by-step install: system library (as admin)
-----------------------------------------------

extract/clone the source to <dir>

For libgpuarray:
::

  cd <dir>
  mkdir Build
  cd Build
  # you can pass -DCMAKE_INSTALL_PREFIX=/path/to/somewhere to install to an alternate location
  cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
  make
  make install
  cd ..

For pygpu:
::

  # This must be done after libgpuarray is installed as per instructions above.
  python setup.py build
  python setup.py install

If you installed libgpuarray in a path that isn't a default one, you
will need to specify where it is. Replace the first line by something
like this:
::

  python setup.py build_ext -L $MY_PREFIX/lib -I $MY_PREFIX/include

If installed globally under Linux (in /usr/local), you might have to run:

.. code-block:: bash

   $ sudo ldconfig

to make the linker know that there are new libraries available.  You
can also reboot the machine to do that.


Step-by-step install: user library
----------------------------------

If you can not or do not want to install it for every user of that
computer, you can install them in your home directory like this:
::

  cd <dir>
  rm -rf ~/.local/lib/libgpuarray* ~/.local/include/gpuarray  ~/.local/lib/python*/site-packages/pygpu*
  rm -rf build Build
  mkdir Build
  cd Build
  cmake .. -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release
  make
  make install
  DEVICE="<test device>" make test

  cd ..

  # Run the following export and add them in your ~/.bashrc file
  export CPATH=$CPATH:~/.local/include
  export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

  python setup.py build
  python setup.py install --user
  cd
  DEVICE="<test device>" python -c "import pygpu;pygpu.test()"

Change ``DEVICE="<test device>"`` to the GPU device you want to use for testing.

Mac-specific instructions
-------------------------

The only supported compiler is the clang version that comes with
Xcode.  Select the appropriate version of Xcode for you version of
macOS.

It might be possible to use a version of gcc built using Homebrew or
MacPorts, but this is untested and unsupported.

If on OS X 10.11 or macOS 10.12 and later and using the system python,
you will have to use a virtualenv to use the python module.  This is
due to a restriction of the new SIP feature about loading libraries.

It appears that on some versions, /usr/local is not in the default
compiler paths so you might need to add ``-L /usr/local/lib -I
/usr/local/include`` to the ``setup.py build`` command or export the
paths like for a custom path install.


Windows-specific instructions
-----------------------------

If you are not comfortable with the command line, you can use the
cmake-gui application to perform the config phase.  It will generate a
Visual Studio solution file for the version installed.  To build the
project open this file (.sln) and run the "Build All" command after
selecting the appropriate build type.

If you prefer a command-line approach, cmake is available as a console
program with the same options as the Unix variant.  You can select the
nmake builder by passing ``-G "NMake Makefiles"`` to cmake.

There is no standard install location on Windows, but you can specify a custom
location by passing ``-DCMAKE_INSTALL_PREFIX=%LIBDIR%`` to cmake. You can then
install using ``cmake --build . --target install`` after ``nmake``.

If you don't have Visual Studio installed, you can get the free `Visual Studio
Community edition <https://www.visualstudio.com/vs/community/>`_,
which has compilation tools for python 3.5 and up. For python 2.7, install
`Microsoft Visual C++ Compiler for Python 2.7 <https://www.microsoft.com/en-ca/download/details.aspx?id=44266>`_.

.. warning::
   While you may get the library to compile using cygwin, this is not
   recommended nor supported.

Running Tests
-------------

.. warning::

   In its current state, the C test suite is woefully incomplete.  It
   will test very basic functionality, but nothing else.  It is
   strongly recommended to run the python test suite to ensure
   everything is ok even if you intend on just using the C library.

To run the C tests, enter the build directory (the one where you ran
cmake), select a target device by exporting DEVICE (or
GPUARRAY_TEST_DEVICE) and run 'make test'.

If you get an error message similar to this one:

::

  Running tests...
  Test project /Users/anakha/ext/gpuarray/Debug
  No tests were found!!!

This means either you don't have check installed or it wasn't found by
the cmake detection script.

To run the python tests, install pygpu, then **move outside** its
directory and run this command:

::

  DEVICE="<test device>" python -c "import pygpu;pygpu.test()"

See the documentation for :py:meth:`pygpu.gpuarray.init` for more
details on the syntax of the device name.

The test script prints the device name of the chosen device so that
you can confirm which device it is running on.

.. note::

   AMD GPUs tend to have really uninformative names, generally being
   only the codename of the architecture the GPU belongs to (e.g.
   'Tahiti').

.. _cmake: https://cmake.org/

.. _clblas: https://github.com/clMathLibraries/clBLAS

.. _clblast: https://github.com/CNugteren/CLBlast

.. _cuda: https://developer.nvidia.com/category/zone/cuda-zone

.. _nccl: https://github.com/NVIDIA/nccl

.. _check: http://check.sourceforge.net/

.. _python: https://python.org/

.. _cython: http://cython.org/

.. _nosetests: https://nose.readthedocs.org/en/latest/

.. _mako: http://www.makotemplates.org/
