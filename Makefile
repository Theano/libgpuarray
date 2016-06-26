rel: install-relc py

-include Makefile.conf

config: Makefile.conf

Makefile.conf:
	@[ ! -f Makefile.conf ] && cp Makefile.conf.tmpl Makefile.conf && echo "\n\n** Adjust the values in Makefile.conf for your system **\n\n" && exit 1

debug: install-debugc py

.PHONY: install-debugc py debug install-relc rel config

Debug:
	mkdir Debug

Debug/Makefile: Debug config
ifndef INSTALL_PREFIX
	(cd Debug && NUM_DEVS=${NUM_DEVS} DEV_NAMES=${DEV_NAMES} cmake .. -DCMAKE_BUILD_TYPE=Debug)
else
	(cd Debug && NUM_DEVS=${NUM_DEVS} DEV_NAMES=${DEV_NAMES} cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX))
endif

debugc: Debug/Makefile
	(cd Debug && make)

test-debugc: debugc
ifndef DEVICE
	(cd Debug && make test)
else
	(cd Debug && DEVICE=${DEVICE} make test)
endif

install-debugc: debugc
	(cd Debug && ${SUDO} make install)

Release:
	mkdir Release

Release/Makefile: Release config
ifndef INSTALL_PREFIX
	(cd Release && NUM_DEVS=${NUM_DEVS} DEV_NAMES=${DEV_NAMES} cmake .. -DCMAKE_BUILD_TYPE=Release)
else
	(cd Release && NUM_DEVS=${NUM_DEVS} DEV_NAMES=${DEV_NAMES} cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX))
endif

relc: Release/Makefile
	(cd Release && make)

test-relc: relc
ifndef DEVICE
	(cd Release && make test)
else
	(cd Release && DEVICE=${DEVICE} make test)
endif

install-relc: relc
	(cd Release && ${SUDO} make install)

py: config
	python setup.py build_ext --inplace
