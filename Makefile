CC=nvcc

all: main main1 main2

main: main.cu threadsinWarp.o 
	$(CC) threadsinWarp.o main.cu -o main

threadsinWarp.o: threadsinWarp.cu threadsinWarp.h
	$(CC) -c threadsinWarp.cu -o threadsinWarp.o


main1: main1.cu threadsinWarp2D.o 
	$(CC) threadsinWarp2D.o main1.cu -o main1

threadsinWarp2D.o: threadsinWarp2D.cu threadsinWarp2D.h
	$(CC) -c threadsinWarp2D.cu -o threadsinWarp2D.o


main2: main2.cu threadsinWarp3D.o 
	$(CC) threadsinWarp3D.o main2.cu -o main2

threadsinWarp3D.o: threadsinWarp3D.cu threadsinWarp3D.h
	$(CC) -c threadsinWarp3D.cu -o threadsinWarp3D.o


clean:
	rm -f main threadsinWarp.o main1 threadsinWarp2D.o main2 threadsinWarp3D.o
