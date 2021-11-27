#ifndef FINALBOIDUTILSKERNEL_H
#define FINALBOIDUTILSKERNEL_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include <vector>

struct bboid
{
    float x;
    float y;
    float dx;
    float dy;
};

void init_boid_structure(std::vector<bboid> &boids, int n, 
                         float *quadVertices, glm::vec2 *translations);
void copy_boid_structure_to_device(std::vector<bboid> &boids, bboid *d_pointer);
void set_initial_boid_position(bboid &boid, float *quadVertices, glm::vec2 &translation);
void move_boid(bboid &boid, glm::mat4 &translation, float x, float y);
void rotate_boid(bboid &boid, glm::mat4 &translation, float angle);
void rotate_boid_and_return(bboid &boid, glm::mat4 &translation, float angle);
void fly_towards_center(bboid &boid, std::vector<bboid> &boids);
float distance(bboid &boid1, bboid &boid2);
void move_logic(bboid &boid, glm::mat4 &translation, float angle);
void keep_within_bounds(bboid &boid, std::vector<bboid> &boids);
void limit_speed(bboid &boid);
void avoid_others(bboid &boid, std::vector<bboid> &boids, unsigned int idx);
void match_velocity(bboid &boid, std::vector<bboid> &boids, unsigned int idx);

#endif