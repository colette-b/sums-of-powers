CC=nvcc
CFLAGS=--compiler-options -Wall

all: main_n44 main_422

aux: basic.cu logging.cc sortedsums.cu

main_n44: aux main_n44.cu 
	$(CC) $(CFLAGS) main_n44.cu -o main_n44

main_422: aux main_422.cu 
	$(CC) $(CFLAGS) main_422.cu -o main_422
