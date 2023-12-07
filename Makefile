CC := nvcc
CFLAGS := -g $(shell sdl2-config --cflags) -ccbin clang-3.8 -arch sm_20 -Wno-deprecated-gpu-targets 
LDFLAGS := $(shell sdl2-config --libs) -lm -lpthread

all: client server 

clean:
	rm -f client
	rm -f server

client: Client.cu board.cu driver.cu
	$(CC) $(CFLAGS) -o client Client.cu board.cu driver.cu $(LDFLAGS) 
server: server.cu driver.cu
	$(CC) $(CFLAGS) -o server server.cu driver.cu $(LDFLAGS) 
