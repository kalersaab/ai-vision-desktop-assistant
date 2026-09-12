CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    TARGET_LIB := build/libfast_vision.dylib
    LDFLAGS := -dynamiclib
else
    TARGET_LIB := build/libfast_vision.so
    LDFLAGS := -shared
endif

all: $(TARGET_LIB)

build:
	mkdir -p build

$(TARGET_LIB): fast_vision.cpp | build
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -rf build

.PHONY: all clean
