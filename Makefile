rel: relc py

-include Makefile.conf

config: Makefile.conf

Makefile.conf:
	@[ ! -f Makefile.conf ] && cp Makefile.conf.tmpl Makefile.conf && echo "\n\n** Adjust the values in Makefile.conf for your system **\n\n" && exit 1

debug: testc py

.PHONY: testc py debug relc rel config

Debug:
	mkdir Debug

Debug/Makefile: Debug config
ifndef INSTALL_DIR
	(cd Debug && cmake .. -DCMAKE_BUILD_TYPE=Debug)
else
	(cd Debug && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX))
endif

testc: Debug/Makefile
	(cd Debug && make && $(SUDO) make install)

Release:
	mkdir Release

Release/Makefile: Release config
ifndef INSTALL_DIR
	(cd Release && cmake .. -DCMAKE_BUILD_TYPE=Release)
else
	(cd Release && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX))
endif

relc: Release/Makefile
	(cd Release && make && $(SUDO) make install)

py: config
	python setup.py build_ext --inplace

