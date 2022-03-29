#include "boid.h"
#include "boids_cpu.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

void cpu::cpu_test(Boid *boids, glm::mat4 *trans, int n,
                   float centering_factor, float visual_range,
                   float margin, float turn_factor,
                   float speed_limit, float min_distance,
                   float avoid_factor, float matching_factor,
                   float mouseX, float mouseY)
{
    for (int index = 0; index < n; index++)
    {
        fly_towards_center(boids, index, n,
                           centering_factor, visual_range);
        avoid_others(boids, index, n, min_distance, avoid_factor);
                
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
}