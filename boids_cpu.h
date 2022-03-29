#ifndef BOIDSCPU_H
#define BOIDSCPU_H
#include "boid.h"
#include "boids_common.h"
#include "glm/glm.hpp"

namespace cpu
{
    void cpu_test(Boid *boids, glm::mat4 *trans, int n,
                  float centering_factor, float visual_range,
                  float margin, float turn_factor,
                  float speed_limit, float min_distance,
                  float avoid_factor, float matching_factor,
                  float mouseX, float mouseY);
}

#endif