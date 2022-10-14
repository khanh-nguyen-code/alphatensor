run: clean
	g++ -std=c++17 -fPIC -O3 -I. -fopenmp main.cpp matmul.cpp -o main.out
	./main.out
clean:
	rm -f main.out
