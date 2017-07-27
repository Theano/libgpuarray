REM Set path for cuda, conda python and cmake
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\Program Files\CMake\bin;C:\lib\cuda\bin

REM Can also set to "Debug", "Release" to go faster
set GPUARRAY_CONFIG="Release"
REM Use spaces to seperate devices
set DEVICES_CUDA=cuda
set DEVICES_OPENCL=

git rev-parse HEAD

REM Clean up previous installs (to make sure no old files are left)
rmdir %WORKSPACE%\lib /s/q
mkdir %WORKSPACE%\lib
rmdir build /s/q
mkdir build

REM Build libgpuarray and run C tests
cd build
cmake .. -DCMAKE_BUILD_TYPE=%GPUARRAY_CONFIG% -G "NMake Makefiles"
nmake
cd ..

set PATH=%PATH%;%WORKSPACE%\lib

REM Add conda gcc toolchain path
set PATH=%PATH%;C:\ProgramData\Miniconda2\Library\mingw-w64\bin;C:\ProgramData\Miniconda2\Library\usr\bin;C:\ProgramData\Miniconda2\Library\bin;C:\ProgramData\Miniconda2\Scripts

REM Build the pygpu modules
python setup.py build_ext --inplace

REM Test pygpu
set test=pygpu
for %%d in (%DEVICES_CUDA%) do (
    echo "Testing pygpu for DEVICE=%%d"
    set DEVICE=%%d
	nosetests --with-xunit --xunit-file=%test%_%DEVICE%_tests.xml pygpu\tests
)
for %%d in (%DEVICES_OPENCL%) do (
    echo "Testing pygpu for DEVICE=%%d"
    set DEVICE=%%d
    nosetests --with-xunit --xunit-file=%test%_%DEVICE%_tests.xml pygpu\tests -e test_blas.py
)
