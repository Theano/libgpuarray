Installation
============

The library is routinely tested on OS X and linux and, less
frequently, on Windows.  The OS most frequently tested are:

 - Debian 6
 - Ubuntu 14.04
 - Mac OS X 10.11
 - Windows 7

It should also work on any decently recent OS not listed here. If you
get an error during the build on your favorite OS, please report it
and we will attempt to fix it.

Requirements
------------

 - cmake >= 3.0 (cmake_).
 - a c99-compliant compiler (or MSVC if on windows).
 - (optional) CUDA >= 6.5 (cuda_).
 - (optional) NVIDIA NCCL (nccl_).
 - (optional) OpenCL runtime.
 - (optional) clBLAS (clblas_).
 - (optional) libcheck (check_) to run the C tests.
 - (optional) python (python_) for the python bindings.
 - (optional) mako (mako_) for development or running the python bindings.
 - (optional) Cython >= 0.21 (cython_) for the python bindings.
 - (optional) nosetests (nosetests_) to run the python tests.

.. note::
   If you have neither an OpenCL runtime or a CUDA runtime, the
   library might still build, but will be rather useless.

.. note::
   We support CUDA GPUs with `compute capability 2.0 (Fermi)
   <https://developer.nvidia.com/cuda-gpus>`_ and up.

.. note::
  In the case you want to build with collective operation support for CUDA,
  you will need CUDA GPUs with `compute capability 3.0 (Kepler)
  <https://developer.nvidia.com/cuda-gpus>`_ and up plus CUDA >= 7.

Download
--------

::

  git clone https://github.com/Theano/libgpuarray.git
  cd libgpuarray

Step-by-step install
--------------------

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


If you can not or do not want to install it for every user of that
computer, you can install them in your home directory like this:
::

  cd <dir>
  rm -rf build Build
  mkdir Build
  cd Build
  cmake .. -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release
  make
  make install
  make test

  cd ..

  # Run the following export and add them in your ~/.bashrc file
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib64/
  export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib64/
  export CPATH=$CPATH:~/.local/include
  export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

  python setup.py build
  python setup.py install --user
  cd
  python -c "import pygpu;pygpu.test()"


Linux-specific instructions
---------------------------

If installed globally (in /usr/local), you might have to run:

.. code-block:: bash

   $ sudo ldconfig

to make the linker know that there are new libraries available.  You
can also reboot the machine to do that.


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

If you don't have Visual Studio installed, you can get the free
Express version from `here <http://www.visualstudio.com/>`_ in the
downloads section (select the "for Windows" edition).

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
  Test project /Users/anakha/ext/gpuarray/Debug
  No tests were found!!!

This means either you don't have check installed or it wasn't found by
the cmake detection script.

To run the python tests, install pygpu, then move outside its
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

.. _cmake: http://cmake.org/

.. _clblas: https://github.com/clMathLibraries/clBLAS

.. _cuda: https://developer.nvidia.com/category/zone/cuda-zone

.. _nccl: https://github.com/NVIDIA/nccl

.. _check: http://check.sourceforge.net/

.. _python: http://python.org/

.. _cython: http://cython.org/

.. _nosetests: http://nose.readthedocs.org/en/latest/

.. _mako: http://www.makotemplates.org/
