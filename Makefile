all: build

build: main.cu utils.cu
	nvcc main.cu -I. -arch=sm_75 -o main

clean: main
	rm main