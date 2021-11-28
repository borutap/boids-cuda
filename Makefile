LDLIBS=-lglut -lGLEW -lGL -lSDL2
all: boids
boids_gpu.o: boids_gpu.cu
	nvcc -c boids_gpu.cu -Xcudafe --diag_suppress=20012
boids_cpu.o: boids_cpu.cu
	nvcc -c boids_cpu.cu -Xcudafe --diag_suppress=20012
boids: boids.cu boids_gpu.o boids_cpu.o
	nvcc boids.cu -Xcudafe --diag_suppress=20012 -o boids boids_gpu.o boids_cpu.o $(LDLIBS)
# ignorowanie flag dot. biblioteki glm typu:
# /usr/include/glm/detail/type_vec2.hpp(94): warning #20012-D: 
# __device__ annotation is ignored on a function("vec") that is
# explicitly defaulted on its first declaration
# nr flagi odkryty z opcji --display-error-number 
clean:
	rm -f *.o boids

.PHONY: all clean