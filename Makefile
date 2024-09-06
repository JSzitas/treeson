all: example

example:
	clang++ -std=c++17 -O0 -ggdb -Wall example.cpp -o example.o; ./example.o

clean:
	rm example.o

