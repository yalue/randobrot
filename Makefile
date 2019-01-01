.PHONY: all clean

all: randobrot

randobrot: randobrot.cpp
	hipcc -o randobrot randobrot.cpp

clean:
	rm -f randobrot
