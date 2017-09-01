cmake -G "%CMAKE_GENERATOR%" ^
      -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
      -DCMAKE_C_FLAGS="-I%LIBRARY_PREFIX%\include" ^
      "%SRC_DIR%"
if errorlevel 1 exit 1
cmake --build . --config Release --target ALL_BUILD
if errorlevel 1 exit 1
cmake --build . --config Release --target install
if errorlevel 1 exit 1
