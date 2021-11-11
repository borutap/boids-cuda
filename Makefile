LDLIBS=-lglut -lGLEW -lGL -lSDL2
all: boids
#boids: boids.cu
#nvcc boids.cu -lglut -lGLEW -lGL -lSDL2 -o boids 
boids: boids.cpp
	g++ -Wall boids.cpp -o boids $(LDLIBS)
clean:
	rm -f *.o boids

.PHONY: all clean