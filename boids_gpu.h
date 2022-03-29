#ifndef BOIDSGPU_H
#define BOIDSGPU_H
#include "boid.h"
#include "boids_common.h"
#include "glm/glm.hpp"

__global__ void kernel_test(Boid *boids, glm::mat4 *trans, int n,
                            float centering_factor, float visual_range,
                            float margin, float turn_factor,
                            float speed_limit, float min_distance,
                            float avoid_factor, float matching_factor,
                            float mouseX, float mouseY);

// Utilities
void copy_boid_structure_to_device(Boid **boids, Boid **d_pointer, int n);
void copy_trans_matrix_to_device(glm::mat4 **mat, glm::mat4 **d_mat, int n);
void copy_trans_matrix_to_host(glm::mat4 **mat, glm::mat4 **d_mat, int n);
void copy_boid_structure_to_host(Boid **boids, Boid **d_pointer, int n);

#endif