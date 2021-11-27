#ifndef BOIDSGPU_H
#define BOIDSGPU_H
#include "glm/glm.hpp"

__device__ __host__ float distance(Boid &boid1, Boid &boid2);
__global__ void kernel_test(Boid *boids, glm::mat4 *trans, int n);
__device__ void fly_towards_center(Boid *boids, int index, int n);
__device__ void keep_within_bounds(Boid *boids, int index);
__device__ void limit_speed(Boid *boids, int index);
__device__ void avoid_others(Boid *boids, int index, int n);
__device__ void match_velocity(Boid *boids, int index, int n);

#endif