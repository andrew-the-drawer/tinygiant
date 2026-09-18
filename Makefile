UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(UNAME),Darwin)
  CC = clang
  CFLAGS = -shared -O3 -mcpu=apple-m1
  LIB = tinygiant/libtinygiant.dylib
else
  CC = gcc
  CFLAGS = -shared -fPIC -O3 -march=armv8.2-a+dotprod
  LIB = tinygiant/libtinygiant.so
endif

SRC = csrc/libtinygiant.c

.PHONY: all clean test

all: $(LIB)

$(LIB): $(SRC)
	$(CC) $(CFLAGS) -lpthread -o $@ $<

test: $(SRC) csrc/test_libtinygiant.c
	$(CC) -O3 -mcpu=apple-m1 -lpthread -lm -Icsrc -o /tmp/test_libtinygiant csrc/test_libtinygiant.c
	/tmp/test_libtinygiant

clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so
