rel: relc testpy

debug: testc testpy

.PHONY: testc testpy debug relc rel

Debug:
	mkdir Debug

Debug/Makefile: Debug
	(cd Debug && cmake .. -DCMAKE_BUILD_TYPE=Debug)

testc: Debug/Makefile
	(cd Debug && make && make install)

Release:
	mkdir Release

Release/Makefile: Release
	(cd Release && cmake .. -DCMAKE_BUILD_TYPE=Release)

relc: Release/Makefile
	(cd Release && make && make install)

testpy:
	python setup.py build_ext --inplace

