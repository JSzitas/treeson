all: example

example:
	clang++ -std=c++17 -O0 -ggdb -Wall -DNO_MULTITHREAD example.cpp -o example.o; ./example.o

m_example:
	clang++ -std=c++17 -I/usr/local/include -L/usr/local/lib -larrow -lparquet -O2 -ggdb -Wall example_multitarg.cpp -o example.o; ./example.o

m_example_n:
	clang++ -std=c++17 -I/usr/local/include -L/usr/local/lib -larrow -lparquet -O2 -ggdb -Wall -DNO_MULTITHREAD example_multitarg.cpp -o example.o; ./example.o

poly_interp:
	clang++ -std=c++17 -O2 -ggdb -Wall test_poly_interp.cpp -o example.o; ./example.o

clean:
	rm example.o

