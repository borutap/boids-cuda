#include "boids_gpu.h"
#include "glm/gtc/matrix_transform.hpp"

__global__ void kernel_test(Boid *boids, glm::mat4 *trans, int n,
                            float centering_factor, float visual_range,
                            float margin, float turn_factor,
                            float speed_limit, float min_distance,
                            float avoid_factor, float matching_factor,
                            float mouseX, float mouseY)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }
        
    fly_towards_center(boids, index, n, centering_factor, visual_range);
    avoid_others(boids, index, n, min_distance, avoid_factor);        
    // sync threads so that velocity is changed before matching
    __syncthreads();
    match_velocity(boids, index, n, matching_factor, visual_range);
    if (mouseX != -2)
        avoid_mouse(boids[index], mouseX, mouseY);
    limit_speed(boids[index], speed_limit);
    keep_within_bounds(boids[index], margin, turn_factor);
    
    Boid &boid = boids[index];
    boid.x += boid.dx;
    boid.y += boid.dy;
    float angle = glm::atan(boid.dy / boid.dx);
    float pi = glm::pi<float>();
    if (boid.dx <= 0)
    {
        angle += pi / 2;
    }        
    else
    {
        angle -= pi / 2;
    }
    auto transformation = glm::translate(glm::mat4(1.0f), glm::vec3(boid.x, boid.y, 0.0f));
    transformation = glm::rotate(transformation, angle, glm::vec3(0.0f, 0.0f, 1.0f));
    trans[index] = transformation;
}

void copy_boid_structure_to_device(Boid **boids, Boid **d_pointer, int n)
{
    size_t size = sizeof(Boid);
    cudaMalloc(d_pointer, n * size);
    cudaMemcpy(*d_pointer, *boids, n * size, cudaMemcpyHostToDevice);
}

void copy_trans_matrix_to_device(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    size_t size = sizeof(glm::mat4);
    cudaMalloc(d_mat, n * size);
    cudaMemcpy(*d_mat, *mat, n * size, cudaMemcpyHostToDevice);
}

void copy_trans_matrix_to_host(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    cudaMemcpy(*mat, *d_mat, n *  sizeof(glm::mat4), cudaMemcpyDeviceToHost);
}

void copy_boid_structure_to_host(Boid **boids, Boid **d_pointer, int n)
{   
    cudaMemcpy(*boids, *d_pointer, n *  sizeof(Boid), cudaMemcpyDeviceToHost);
}