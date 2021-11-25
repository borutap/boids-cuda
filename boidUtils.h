#ifndef boidUtils_H
#define boidUtils_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "myMath.h"
#include <iostream>
#include <vector>

const float SPEED_FACTOR_X = 0.001f;
const float SPEED_FACTOR_Y = 0.001f;
const float MAX_SPEED = 0.01f;
const float VIEW_DEGREE = glm::pi<float>() / 3; // r 
const float MAX_DISTANCE = 0.2f; // d
const float D_MIN = MAX_DISTANCE / 2.5f;
const float WEIGHT_SPEED = 0.1f;
const float WEIGHT_DISTANCE = 0.15f;
const float WEIGHT_MIN = 0.15f;
const float WEIGHT_RANDOM = 0.1f;

struct boid
{
    glm::vec2 pos;
    glm::vec2 speed;
    int neighbor_count;
    glm::vec2 neighbors_mean_speed;    
    float neighbors_mean_distance;
};

glm::vec2 get_initial_boid_position(float *quadVertices, glm::vec2 &translation);
void neighbor_search_action(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations);
void move_boid(glm::mat4 &matrix, glm::vec2 &origin_translation, float x, float y);
void init_boids(std::vector<boid> &boids, float *quadVertices, glm::vec2 *translations, int n);
void on_update(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations);
void transform_boid(glm::mat4 &matrix, glm::mat4 &move_translation,
                    glm::mat4 &rotation, glm::vec2 &origin_translation,
                    glm::vec2 &speed, glm::vec2 &pos);

#endif