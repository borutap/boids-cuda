#include "myMath.h"

bool is_neighbor_distance(float distance, float max_distance)
{
    if (distance < max_distance)
    {
        return true;   
    }     
    return false;
}

float boid_distance_to_neighbor(glm::vec2 &neighbor_pos, glm::vec2 &boid_pos)
{

    auto ret = glm::sqrt(glm::pow(neighbor_pos.x - boid_pos.x, 2) +
                     glm::pow(neighbor_pos.y - boid_pos.y, 2));
    return ret;
}

// all angles are in radians
float move_angle(float y_speed, float x_speed)
{
    if (glm::abs(x_speed - 0.001f) < 0.01f)
        return 1.5705f; // pi / 2
    return glm::atan(y_speed / x_speed);
}

float boid_angle_to_neighbor(glm::vec2 &neighbor_pos, glm::vec2 &boid_pos)
{
    if (glm::abs(neighbor_pos.x - boid_pos.x) < 0.01f)
        return 1.5705f; // pi / 2
    auto ret = glm::atan((neighbor_pos.y - boid_pos.y) / (neighbor_pos.x - boid_pos.x));
    return ret;
}

bool is_neighbor_angle(float move_angle, float angle_to_neighbor, float vision_angle)
{
    if (glm::abs(angle_to_neighbor - move_angle) < vision_angle)
        return true;
    return false;
}