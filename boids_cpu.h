#ifndef BOIDSCPU_H
#define BOIDSCPU_H
#include "glm/glm.hpp"

namespace cpu
{
    float distance(Boid &boid1, Boid &boid2);
    void cpu_test(Boid *boids, glm::mat4 *trans, int n,
                  float centering_factor, float visual_range,
                  float margin, float turn_factor,
                  float speed_limit, float min_distance,
                  float avoid_factor, float matching_factor);
    void fly_towards_center(Boid *boids, int index, int n,
                            float centering_factor,
                            float visual_range);
    void keep_within_bounds(Boid *boids, int index,
                            float margin, float turn_factor);
    void limit_speed(Boid *boids, int index, float speed_limit);
    void avoid_others(Boid *boids, int index, int n,
                      float min_distance, float avoid_factor);
    void match_velocity(Boid *boids, int index, int n,
                        float matching_factor,
                        float visual_range);
}

#endif