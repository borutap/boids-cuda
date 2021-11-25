#ifndef MYMATH_H
#define MYMATH_H

#include <glm/glm.hpp>

bool is_neighbor_distance(float distance, float max_distance);
float boid_distance_to_neighbor(glm::vec2 &neighbor_pos, glm::vec2 &boid_pos);
float move_angle(float y_speed, float x_speed);
float boid_angle_to_neighbor(glm::vec2 &neighbor_pos, glm::vec2 &boid_pos);
bool is_neighbor_angle(float move_angle, float angle_to_neighbor, float vision_angle);

#endif