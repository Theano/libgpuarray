del bld
mkdir bld
cd bld
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
