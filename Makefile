LDLIBS=-lglut -lGLEW -lGL -lSDL2
all: boids
#boids: boids.cu
#nvcc boids.cu -lglut -lGLEW -lGL -lSDL2 -o boids 
boids: boids.cpp
	g++ -Wall -g boids.cpp finalBoidUtils.cpp -o boids $(LDLIBS)
testt: testt.cu
	nvcc testt.cu -Xcudafe --diag_suppress=20012 -o testt $(LDLIBS)

# boids_common.o: boids_common.cu
# 	nvcc -c boids_common.cu
boids_gpu.o: boids_gpu.cu
	nvcc -c boids_gpu.cu -Xcudafe --diag_suppress=20012
boids_cpu.o: boids_cpu.cu
	nvcc -c boids_cpu.cu -Xcudafe --diag_suppress=20012
# kernel.o: kernel.cu
# 	nvcc -c kernel.cu
# main: main.o gpu.o
# 	nvcc -o main main.o gpu.o
kernel: boids.cu boids_gpu.o boids_cpu.o
	nvcc boids.cu -Xcudafe --diag_suppress=20012 -o kernel boids_gpu.o boids_cpu.o $(LDLIBS)
# ignorowanie flag dot. biblioteki glm typu:
# /usr/include/glm/detail/type_vec2.hpp(94): warning #20012-D: 
# __device__ annotation is ignored on a function("vec") that is
# explicitly defaulted on its first declaration
# nr flagi odkryty z opcji --display-error-number 
clean:
	rm -f *.o boids

.PHONY: all clean