REM Set path for conda python and cmake
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\Program Files\CMake\bin

# Can also set to "Debug", "Release" to go faster
set GPUARRAY_CONFIG="Release"
# Set these to " " to disable (empty doesn't work)
set DEVICES_CUDA="cuda" # for multiple devices use "cuda0 cuda1"
set DEVICES_OPENCL=""

git rev-parse HEAD

# Build libgpuarray and run C tests
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=%GPUARRAY_CONFIG% -G "NMake Makefiles"
nmake
cd ..

REM Export paths
set LIBDIR=%WORKSPACE%\local
set PATH=%PATH%;%LIBDIR%\lib;C:\lib\cuda\bin

REM Clean up previous installs (to make sure no old files are left)
rmdir %LIBDIR% /s/q
mkdir %LIBDIR%

# Test on different devices
(for %%dev in (%DEVICES_CUDA%) do (
    echo "Testing libgpuarray for DEVICE=%%dev"
    cd build
    set DEVICE=%%dev
    make test
    cd ..
))

(for %%dev in (%DEVICES_OPENCL%) do (
    echo "Testing libgpuarray for DEVICE=%%dev"
    cd build
    set DEVICE=%%dev
    make test
    cd ..
))

REM Set conda python path
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\ProgramData\Miniconda2\Library\mingw-w64\bin;C:\ProgramData\Miniconda2\Library\usr\bin;C:\ProgramData\Miniconda2\Library\bin;C:\ProgramData\Miniconda2\Scripts

REM Build the pygpu modules
python setup.py build_ext --inplace

# Test it
set test=pygpu
(for %%dev in (%DEVICES_CUDA%) do (
    echo "Testing pygpu for DEVICE=%%dev"
    set DEVICE=%%dev
    nosetests --with-xunit --xunit-file=%test%_%DEVICE%_tests.xml pygpu\tests
))
(for %%dev in (%DEVICES_OPENCL%) do (
    echo "Testing pygpu for DEVICE=%%dev"
    set DEVICE=%%dev
    nosetests --with-xunit --xunit-file=%test%_%DEVICE%_tests.xml pygpu\tests
))
