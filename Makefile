all: example

example:
	clang++ -std=c++17 -O0 -ggdb -Wall example.cpp -o example.o; ./example.o

m_example:
	clang++ -std=c++17 -O2 -ggdb -Wall example_multitarg.cpp -o example.o; ./example.o

m_example_n:
	clang++ -std=c++17 -O2 -ggdb -Wall -DNO_MULTITHREAD example_multitarg.cpp -o example.o; ./example.o

clean:
	rm example.o

