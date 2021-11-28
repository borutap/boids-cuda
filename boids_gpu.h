#ifndef BOIDSGPU_H
#define BOIDSGPU_H
#include "glm/glm.hpp"

__device__ float distance(Boid &boid1, Boid &boid2);
__global__ void kernel_test(Boid *boids, glm::mat4 *trans, int n,
                            float centering_factor, float visual_range,
                            float margin, float turn_factor,
                            float speed_limit, float min_distance,
                            float avoid_factor, float matching_factor);
__device__ void fly_towards_center(Boid *boids, int index, int n,
                                   float centering_factor,
                                   float visual_range);
__device__ void keep_within_bounds(Boid *boids, int index,
                                   float margin, float turn_factor);
__device__ void limit_speed(Boid *boids, int index, float speed_limit);
__device__ void avoid_others(Boid *boids, int index, int n,
                             float min_distance, float avoid_factor);
__device__ void match_velocity(Boid *boids, int index, int n,
                               float matching_factor,
                               float visual_range);

#endif