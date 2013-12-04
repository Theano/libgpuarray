Installation
============

The library is routinely tested on OS X and linux and, less
frequently, on Windows.  The OS most frequently tested are:

 - Mac OS X 10.7
 - Fedora Core 14
 - Ubuntu 12.04
 - Mac OS X 10.9
 - Windows 7

It should also work on any decently recent OS not listed here. If you
get an error during the build on your favorite OS, please report it
and we will attempt to fix it.

Requirements
------------

 - cmake >= 2.8 (cmake_).
 - a c99-compliant compiler (or MSVC if on windows).

 - (optional) CUDA >= 4.0 (cuda_).
 - (optional) OpenCL runtime.
 - (optional) clBLAS (clblas_).
 - (optional) libcheck (check_) to run the C tests.
 - (optional) python (python_) for the python bindings.
 - (optional) Cython >= 0.19 (cython_) for the python bindings.
 - (optional) nosetests (nosetests_) to run the python tests.

.. note::
   If you have neither an OpenCL runtime or a CUDA runtime, the
   library might still build, but will be rather useless.

Step-by-step install
--------------------

extract/clone the source to <dir>

For libcompyte:
::

  cd <dir>
  mkdir Build
  cd Build
  # you can pass -DCMAKE_INSTALL_PREFIX=/path/to/somewhere to install to an alternate location
  cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
  make
  make install

For pygpu:
::

  # This must be done after libcompyte is installed as per instructions above.
  python setup.py build
  python setup.py install

Mac-specific instructions
-------------------------

To get the compiler you need to install Xcode which is available for
free from the App Store.  Don't forget to install the command-line
tools afterwards.

On Xcode 4.x these are installed by going to the download tab of the
preferences window and selecting the "Command-line Tools" download.

If you have Xcode 5, ensure you update to 5.0.2 or later.  Prior
versions will not look in /usr/local for includes or libraries and
this will cause a lot of errors.  You can update by using the
"Software Update..." function of the Apple menu or by running
'xcode-select --install' on the command line.

It might be possible to use a version of gcc built using Homebrew or
MacPorts, but this is untested and unsupported.

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

Since there is no standard install location on Windows, there is no
install step.  It is up to you to copy the headers and libraries to an
appropriate place.

If you don't have Visual Studio installed, you can get the free Express version from `here <http://www.visualstudio.com/>`_ in the downloads section (select the "for Windows" edition).

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
cmake) and run 'make test'.  It will run using the first OpenCL and
the first CUDA device it finds skipping these if the corresponding
backend wasn't built.

If you get an error message similar to this one:

::

  Running tests...
  Test project /Users/anakha/ext/compyte/Debug
  No tests were found!!!

This means either you don't have check installed or it wasn't found by
the cmake detection script.

To run the python tests run nosetests in the pygpu subdirectory.  By
default it will attempt to use 'opencl0:0' as the compute device but
you can override this by setting the DEVICE or COMPYTE_DEVICE
environement variable, with COMPYTE_DEVICE having priority, if set.
The format for the device string is '<backend name><device id>'.
Possible backend names are 'cuda' and 'opencl'.

For 'cuda' possible device ids are from 0 to the number of cuda
devices.

For 'opencl' the devices id are of this format '<platform
number>:<device number>'.  Both start at 0 and go up to the number of
platforms/devices available.  There is no fixed order for the devices,
but the order on a single machine should be stable across runs.

The test script prints the device name of the chosen device so that
you can confirm which device it is running on.

.. note::

   AMD GPUs tend to have really uninformative names, generally being
   only the codename of the architecture the GPU belongs to (e.g.
   'Tahiti').

.. _cmake: http://cmake.org/

.. _clblas: https://github.com/clMathLibraries/clBLAS

.. _cuda: https://developer.nvidia.com/category/zone/cuda-zone

.. _check: http://check.sourceforge.net/

.. _python: http://python.org/

.. _cython: http://cython.org/

.. _nosetests: http://nose.readthedocs.org/en/latest/
