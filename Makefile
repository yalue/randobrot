.PHONY: all clean

all: randobrot

randobrot: randobrot.cpp hip_rng.cpp hip_rng.h
	hipcc -g -Wall -Werror -O3 -o randobrot randobrot.cpp hip_rng.cpp

clean:
	rm -f randobrot
