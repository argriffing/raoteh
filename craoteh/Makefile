
all: sampler

sampler: sampler.o
	gcc -L/usr/local/lib sampler.o -lgsl -lgslcblas -lm -o sampler

sampler.o:
	gcc -Wall -I/user/local/include -c sampler.c 

clean:
	rm -rf *.o sampler
