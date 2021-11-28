#ifndef BOIDSCPU_H
#define BOIDSCPU_H
#include "glm/glm.hpp"

namespace cpu
{
    float distance(Boid &boid1, Boid &boid2);
    void cpu_test(Boid *boids, glm::mat4 *trans, int n);
    void fly_towards_center(Boid *boids, int index, int n);
    void keep_within_bounds(Boid *boids, int index);
    void limit_speed(Boid *boids, int index);
    void avoid_others(Boid *boids, int index, int n);
    void match_velocity(Boid *boids, int index, int n);
}


#endif