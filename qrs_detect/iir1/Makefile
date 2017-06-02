.PHONY: clean deps cleanall

MODULES=pyiir1.so
PYTHON=python
SRC=pyiir1.c config.h butterworth.h butterworth.c

all: $(MODULES)


deps: c/Makefile

c/Makefile:
	cd c && ./autogen.sh
	cd c && ./configure --enable-static

%.so: %.c c/libiir.la $(SRC)
	$(PYTHON) setup.py build
	cp build/lib.*/$@ .
	cp $@ ../

c/libiir.la: deps
	make -C ./c

cleanall: clean
	rm -f c/Makefile
	rm -f c/libiir.la

clean:
	rm -rf build
	rm -f $(MODULES)
	

