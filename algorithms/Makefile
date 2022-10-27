run: clean
	g++ \
		-std=c++17 -fPIC -O3 \
		-I. \
		-I/home/khanh/miniconda3/envs/test/include/ \
		-L/home/khanh/miniconda3/envs/test/lib/ \
		main.cpp matmul.cpp -o main.out \
		-lopenblas -lrt
	./main.out
clean:
	rm -f main.out
