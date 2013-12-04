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

On Xcode 5.x these are installed by stating the GUI app once, then
running 'xcode-select --install'.  There are tools installed by just
running the GUI app but they have incomplete search path and you will
encounter a lot of problems with them.

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

.. _cmake: http://cmake.org/

.. _clblas: https://github.com/clMathLibraries/clBLAS

.. _cuda: https://developer.nvidia.com/category/zone/cuda-zone

.. _check: http://check.sourceforge.net/

.. _python: http://python.org/

.. _cython: http://cython.org/

.. _nosetests: http://nose.readthedocs.org/en/latest/
