all: example

example:
	clang++ -std=c++17 -O2 -ggdb -Wall examples/example.cpp -o example.o; ./example.o

m_example:
	clang++ -std=c++17 -I/usr/local/include -L/usr/local/lib -larrow -lparquet -O2  -ggdb -Wall examples/example_multitarg.cpp -o example.o; ./example.o

m_example_n:
	clang++ -std=c++17 -I/usr/local/include -L/usr/local/lib -larrow -lparquet -O2 -ggdb -Wall examples/example_multitarg.cpp -o example.o; ./example.o

profile:
	clang++ -std=c++17 -O0 -g -Wall examples/example_multitarg.cpp -o profile.o; ./profile.o

profile_m:
	clang++ -std=c++17 -O0 -g -Wall examples/example_multitarg.cpp -o profile_m.o; ./profile_m.o

profile_big:
	clang++ -std=c++17 -O0 -g -Wall -I/usr/local/include -L/usr/local/lib -larrow -lparquet examples/example_multitarg.cpp -o profile_b.o; ./profile_b.o

clean:
	rm *.o *.bin

