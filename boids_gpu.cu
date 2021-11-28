#include "boids_common.h"
#include "boids_gpu.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

__global__ void kernel_test(Boid *boids, glm::mat4 *trans, int n,
                            float centering_factor, float visual_range,
                            float margin, float turn_factor,
                            float speed_limit, float min_distance,
                            float avoid_factor, float matching_factor)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }
        
    fly_towards_center(boids, index, n, centering_factor, visual_range);
    avoid_others(boids, index, n, min_distance, avoid_factor);
    // zeby zmiana predkosci (dx, dy) w avoid_others 
    // nie zaburzyla sredniej liczonej
    // w innym watku w match_velocity
    __syncthreads();
    match_velocity(boids, index, n, matching_factor, visual_range);
    limit_speed(boids, index, speed_limit);
    keep_within_bounds(boids, index, margin, turn_factor);
    // __syncthreads(); // wczesniej nie zmieniamy x, y
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
    // printf("boids[%d].x = %f\n", index, boid.x);
    // cout << "boid.x = " << boid.x << endl;
    // cout << "boid.y = " << boid.y << endl;
    // cout << "boid.dx = " << boid.dx << endl;
    // cout << "boid.dy = " << boid.dy << endl;
}

__device__ void fly_towards_center(Boid *boids, int index, int n,
                                   float centering_factor,
                                   float visual_range)
{    
    Boid &boid = boids[index];
    // const float centering_factor = 0.002f; // adjust velocity by this %
    // const float visual_range = 0.05f;

    float centerX = 0.0f;
    float centerY = 0.0f;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < n; i++)
    {
        Boid &other = boids[i];
        if (distance(boid, other) < visual_range)
        {
            centerX += other.x;
            centerY += other.y;
            num_neighbors += 1;
        }
    }

    if (num_neighbors)
    {
        centerX = centerX / num_neighbors;
        centerY = centerY / num_neighbors;

        boid.dx += (centerX - boid.x) * centering_factor;
        boid.dy += (centerY - boid.y) * centering_factor;
    }
}

__device__ float distance(Boid &boid1, Boid &boid2)
{
    return glm::sqrt(
        (boid1.x - boid2.x) * (boid1.x - boid2.x) +
            (boid1.y - boid2.y) * (boid1.y - boid2.y)
    );
}

__device__ void keep_within_bounds(Boid *boids, int index,
                                   float margin, float turn_factor)
{
    Boid &boid = boids[index];
    // const float margin = 0.1f;
    // const float turn_factor = 1.0f / 2000;

    if (boid.x < -1.0f + margin)
        boid.dx += turn_factor;    

    if (boid.x > 1.0f - margin)
        boid.dx -= turn_factor;

    if (boid.y < -1.0f + margin)
        boid.dy += turn_factor;    

    if (boid.y > 1.0f - margin)
        boid.dy -= turn_factor;
}

__device__ void limit_speed(Boid *boids, int index, float speed_limit)
{
    Boid &boid = boids[index];
    // const float speed_limit = 0.005f;

    float speed = glm::sqrt(boid.dx * boid.dx + boid.dy * boid.dy);
    if (speed <= speed_limit)
        return;
    boid.dx = (boid.dx / speed) * speed_limit;
    boid.dy = (boid.dy / speed) * speed_limit;
}

__device__ void avoid_others(Boid *boids, int index, int n,
                             float min_distance, float avoid_factor)
{
    Boid &boid = boids[index];
    // const float min_distance = 0.014f; // The distance to stay away from other boids
    // const float avoid_factor = 0.05f; // Adjust velocity by this %
    float moveX = 0;
    float moveY = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        if (i == index)
            continue;
        Boid &other = boids[i];        
        if (distance(boid, other) < min_distance)
        {
            moveX += boid.x - other.x;
            moveY += boid.y - other.y;
        }
    } 
    
    boid.dx += moveX * avoid_factor;
    boid.dy += moveY * avoid_factor;
}

__device__ void match_velocity(Boid *boids, int index, int n,
                               float matching_factor,
                               float visual_range)
{
    Boid &boid = boids[index];
    // const float matching_factor = 0.05f;
    // const float visual_range = 0.05f; // TODO - to do glob. zmiennej

    float avgDX = boid.dx;
    float avgDY = boid.dy;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < n; i++)
    {
        if (i == index)
            continue;
        Boid &other = boids[i];
        if (distance(boid, other) < visual_range)
        {
            avgDX += other.dx;
            avgDY += other.dy;
            num_neighbors += 1;
        }
    }
    // zeby zmiana dx, dy nizej nie zaburzyla
    // sredniej liczonej w innym watku wyzej
    __syncthreads();    
    if (num_neighbors)
    {
        avgDX = avgDX / num_neighbors;
        avgDY = avgDY / num_neighbors;

        boid.dx += (avgDX - boid.dx) * matching_factor;
        boid.dy += (avgDY - boid.dy) * matching_factor;
    }
}