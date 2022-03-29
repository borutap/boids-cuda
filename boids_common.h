#ifndef PARTICLECOMMON_H
#define PARTICLECOMMON_H
#include "boid.h"

__device__ __host__ float distance(Boid &boid1, Boid &boid2);
__device__ __host__ void fly_towards_center(Boid *boids, int index, int n,
                                            float centering_factor,
                                            float visual_range);
__device__ __host__ void keep_within_bounds(Boid& boid, float margin, float turn_factor);
__device__ __host__ void limit_speed(Boid& boid, float speed_limit);
__device__ __host__ void avoid_others(Boid *boids, int index, int n,
                                      float min_distance, float avoid_factor);
__device__ __host__ void avoid_mouse(Boid& boid, float mouseX, float mouseY);
__device__ __host__ void match_velocity(Boid *boids, int index, int n,
                               float matching_factor,
                               float visual_range);

#endif