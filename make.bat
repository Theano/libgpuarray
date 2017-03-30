REM This helps repetitive builds on windows
REM It needs the compiler you want to use to be available in the shell
REM and it will build a release version

del bld
mkdir bld
cd bld
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
