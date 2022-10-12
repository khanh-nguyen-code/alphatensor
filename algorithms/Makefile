run: build
	g++ -std=c++17 -fPIC -O3 -I. main.cpp *.o -o main
	./main
	rm -f main *.o

build:
	g++ -std=c++17 -fPIC -c -O3 -I. matmul.cpp -o matmul.o
