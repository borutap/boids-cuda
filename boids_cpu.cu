#include "boids_common.h"
#include "boids_cpu.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

void cpu::cpu_test(Boid *boids, glm::mat4 *trans, int n)
{
    for (int index = 0; index < n; index++)
    {
        fly_towards_center(boids, index, n);
        avoid_others(boids, index, n);
                
        match_velocity(boids, index, n);
        limit_speed(boids, index);
        keep_within_bounds(boids, index);    
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
}

void cpu::fly_towards_center(Boid *boids, int index, int n)
{    
    Boid &boid = boids[index];
    const float centering_factor = 0.002f; // adjust velocity by this %
    const float visual_range = 0.05f;

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

float cpu::distance(Boid &boid1, Boid &boid2)
{
    return glm::sqrt(
        (boid1.x - boid2.x) * (boid1.x - boid2.x) +
            (boid1.y - boid2.y) * (boid1.y - boid2.y)
    );
}

void cpu::keep_within_bounds(Boid *boids, int index)
{
    Boid &boid = boids[index];
    const float margin = 0.1f;
    const float turn_factor = 1.0f / 2000;

    if (boid.x < -1.0f + margin)
        boid.dx += turn_factor;    

    if (boid.x > 1.0f - margin)
        boid.dx -= turn_factor;

    if (boid.y < -1.0f + margin)
        boid.dy += turn_factor;    

    if (boid.y > 1.0f - margin)
        boid.dy -= turn_factor;
}

void cpu::limit_speed(Boid *boids, int index)
{
    Boid &boid = boids[index];
    const float speed_limit = 0.005f;

    float speed = glm::sqrt(boid.dx * boid.dx + boid.dy * boid.dy);
    if (speed <= speed_limit)
        return;
    boid.dx = (boid.dx / speed) * speed_limit;
    boid.dy = (boid.dy / speed) * speed_limit;
}

void cpu::avoid_others(Boid *boids, int index, int n)
{
    Boid &boid = boids[index];
    const float min_distance = 0.014f; // The distance to stay away from other boids
    const float avoid_factor = 0.05f; // Adjust velocity by this %
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

void cpu::match_velocity(Boid *boids, int index, int n)
{
    Boid &boid = boids[index];
    const float matching_factor = 0.05f;
    const float visual_range = 0.05f; // TODO - to do glob. zmiennej

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