LDLIBS=-lGLEW -lGL -lSDL2
all: boids
# diag_suppress=20012 suppresses warning from glm library
# thanks to -dc flag we can use functions from other files
boids_common.o: boids_common.cu
	nvcc -dc -c boids_common.cu -Xcudafe --diag_suppress=20012
boids_gpu.o: boids_gpu.cu boids_common.o
	nvcc -dc -c boids_gpu.cu -Xcudafe --diag_suppress=20012
boids_cpu.o: boids_cpu.cu
	nvcc -dc -c boids_cpu.cu -Xcudafe --diag_suppress=20012
parameters.o: parameters.cu
	nvcc -c parameters.cu
logger.o: logger.cu
	nvcc -c logger.cu
boids: boids.cu boids_gpu.o boids_cpu.o parameters.o logger.o
	nvcc boids.cu -Xcudafe --diag_suppress=20012 -o boids boids_gpu.o boids_common.o boids_cpu.o parameters.o logger.o $(LDLIBS)
clean:
	rm -f *.o boids

.PHONY: all clean