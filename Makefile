all: build

build: main.cu helpers.h
	nvcc main.cu -I. -I/usr/local/cuda/include -lcublasLt -arch=sm_75 -o main

clean: main
	rm main