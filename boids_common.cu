#include "boids_common.h"
#include "glm/glm.hpp"

__device__ __host__ void fly_towards_center(Boid *boids, int index, int n,
                                   float centering_factor,
                                   float visual_range)
{    
    Boid &boid = boids[index];
    // centering_factor = adjust velocity by this %

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

__device__ __host__ float distance(Boid &boid1, Boid &boid2)
{
    return glm::sqrt(
        (boid1.x - boid2.x) * (boid1.x - boid2.x) +
            (boid1.y - boid2.y) * (boid1.y - boid2.y)
    );
}

__device__ __host__ void keep_within_bounds(Boid &boid, float margin, float turn_factor)
{
    if (boid.x < -1.0f + margin)
        boid.dx += turn_factor;    

    if (boid.x > 1.0f - margin)
        boid.dx -= turn_factor;

    if (boid.y < -1.0f + margin)
        boid.dy += turn_factor;    

    if (boid.y > 1.0f - margin)
        boid.dy -= turn_factor;
}

__device__ __host__ void limit_speed(Boid &boid, float speed_limit)
{
    float speed = glm::sqrt(boid.dx * boid.dx + boid.dy * boid.dy);
    if (speed <= speed_limit)
        return;
    boid.dx = (boid.dx / speed) * speed_limit;
    boid.dy = (boid.dy / speed) * speed_limit;
}

__device__ __host__ void avoid_others(Boid *boids, int index, int n,
                             float min_distance, float avoid_factor)
{
    Boid &boid = boids[index];
    // min_distance = The distance to stay away from other boids
    // avoid_factor = Adjust velocity by this %
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

__device__ __host__ void avoid_mouse(Boid &boid, float mouseX, float mouseY)
{
    // min_distance = The distance to stay away from other boids
    // avoid_factor = Adjust velocity by this %
    float min_distance = 0.07;
    float avoid_factor = 0.5;    

    if (glm::sqrt(
            (boid.x - mouseX) * (boid.x - mouseX) +
            (boid.y - mouseY) * (boid.y - mouseY)) < min_distance)
    {
        boid.dx += boid.x - mouseX * avoid_factor;
        boid.dy += boid.y - mouseY * avoid_factor;        
    }        
}

__device__ __host__ void match_velocity(Boid *boids, int index, int n,
                               float matching_factor,
                               float visual_range)
{
    Boid &boid = boids[index];

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
     
    if (num_neighbors)
    {
        avgDX = avgDX / num_neighbors;
        avgDY = avgDY / num_neighbors;

        boid.dx += (avgDX - boid.dx) * matching_factor;
        boid.dy += (avgDY - boid.dy) * matching_factor;
    }
}