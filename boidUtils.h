#ifndef boidUtils_H
#define boidUtils_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "myMath.h"
#include <iostream>
#include <vector>

// const float SPEED_FACTOR_X = 0.0000001f;
// const float SPEED_FACTOR_Y = 0.0000001f;
// const float MAX_SPEED = 0.000001f;
// // const float VIEW_DEGREE = glm::pi<float>() / 3; // r 
// const float VIEW_DEGREE = 120.0f; // r 
// const float MAX_DISTANCE = 0.4f; // d
// const float D_MIN = 0.5f;
// const float WEIGHT_SPEED = 0.04f;
// const float WEIGHT_DISTANCE = 0.15f;
// const float WEIGHT_MIN = 0.05f;
// const float WEIGHT_RANDOM = 0.2f;
const float SPEED_FACTOR_X = 0.0004f;
const float SPEED_FACTOR_Y = 0.0004f;
const float MAX_SPEED = 0.004f;
// const float VIEW_DEGREE = glm::pi<float>() / 3; // r 
const float VIEW_DEGREE = 120.0f; // r 
const float MAX_DISTANCE = 50.0f; // d
const float D_MIN = 20.0f;
const float WEIGHT_SPEED = 0.1f;
const float WEIGHT_DISTANCE = 0.15f;
const float WEIGHT_MIN = 0.15f;
const float WEIGHT_RANDOM = 0.1f;

struct boid
{
    float pos_x;
    float pos_y;
    float speed_x;
    float speed_y;
    int neighbor_count;
    float neighbors_mean_speed_x;    
    float neighbors_mean_speed_y;    
    float neighbors_mean_distance;
};

void myMain(std::vector<boid> &boids, int n);
glm::vec2 get_initial_boid_position(float *quadVertices, glm::vec2 &translation);
void neighbor_search_action(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations);
void move_boid(glm::mat4 &matrix, glm::vec2 &origin_translation, float x, float y);
void init_boids(std::vector<boid> &boids, float *quadVertices, glm::vec2 *translations, int n,
                int scr_width, int scr_height);
void on_update(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations);
void transform_boid(glm::mat4 &matrix, glm::mat4 &move_translation,
                    glm::mat4 &rotation, glm::vec2 &origin_translation,
                    boid &boid);
glm::vec2 convertToBig(glm::vec2 &original, int scr_width, int scr_height);
void convertToBig(boid &boid, int scr_width, int scr_height);                    
void convertToSmall(boid &boid, int scr_width, int scr_height);

#endif