#include "finalBoidUtilsKernel.h"

void init_boid_structure(std::vector<bboid> &boids, int n, 
                         float *quadVertices, glm::vec2 *translations)
{
    for (int i = 0; i < n; i++)
    {
        boids[i].dx = glm::linearRand(-0.5f, 0.5f) / 100;
        boids[i].dy = glm::linearRand(-0.5f, 0.5f) / 100;
        set_initial_boid_position(boids[i], quadVertices, translations[i]);        
    }
}

void copy_boid_structure_to_device(std::vector<bboid> &boids, bboid *d_pointer)
{
    size_t size = sizeof(bboid);  
    cudaMalloc(&d_pointer, boids.size() * size);    
    for (auto it = boids.begin(); it != boids.end(); it++)
    {
        bboid *src = &(*it);        
        cudaMemcpy(d_pointer, src, size, cudaMemcpyHostToDevice);
        d_pointer += size;
    }    
}

void set_initial_boid_position(bboid &boid, float *quadVertices, glm::vec2 &translation)
{
    float origin_x = 0.0f;
    float origin_y = (quadVertices[6] + quadVertices[1]) / 2;
    boid.x = origin_x + translation.x;
    boid.y = origin_y + translation.y;    
}

void move_boid(bboid &boid, glm::mat4 &translation, float x, float y)
{    
    boid.x = x;
    boid.y = y;
    translation = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, 0.0f));
}

void rotate_boid(bboid &boid, glm::mat4 &translation, float angle)
{    
    translation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
}

void rotate_boid_and_return(bboid &boid, glm::mat4 &translation, float angle)
{
    translation = glm::rotate(translation, angle, glm::vec3(0.0f, 0.0f, 1.0f));
}

void move_logic(bboid &boid, glm::mat4 &translation, float angle)
{
    // auto move1 = glm::translate(glm::mat4(1.0f), glm::vec3(-boid.x, -boid.y, 0.0f));
    // auto move1 = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
    auto rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
    auto move2 = glm::translate(glm::mat4(1.0f), glm::vec3(boid.x, boid.y, 0.0f));
    translation = move2 * rotation;// * move1;
    // translation = glm::translate(translation, glm::vec3(-boid.x, -boid.y, 0.0f));
}

void keep_within_bounds(bboid &boid, std::vector<bboid> &boids)
{
    const float margin = 0.15f;
    const float turn_factor = 1.0f / 1000;

    if (boid.x < -1.0f + margin)
        boid.dx += turn_factor;    

    if (boid.x > 1.0f - margin)
        boid.dx -= turn_factor;

    if (boid.y < -1.0f + margin)
        boid.dy += turn_factor;    

    if (boid.y > 1.0f - margin)
        boid.dy -= turn_factor;
}

void avoid_others(bboid &boid, std::vector<bboid> &boids, unsigned int idx)
{
    const float min_distance = 0.01f; // The distance to stay away from other boids
    const float avoid_factor = 0.05f; // Adjust velocity by this %
    float moveX = 0;
    float moveY = 0;
    for (unsigned int i = 0; i < boids.size(); i++)
    {
        if (i == idx)
            continue;
        bboid &other = boids[i];        
        if (distance(boid, other) < min_distance)
        {
            moveX += boid.x - other.x;
            moveY += boid.y - other.y;
        }
    } 
    
    boid.dx += moveX * avoid_factor;
    boid.dy += moveY * avoid_factor;
}

void match_velocity(bboid &boid, std::vector<bboid> &boids, unsigned int idx)
{
    const float matching_factor = 0.05f;
    const float visual_range = 0.08f; // TODO - to do glob. zmiennej

    float avgDX = 0.0f;
    float avgDY = 0.0f;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < boids.size(); i++)
    {
        if (i == idx)
            continue;
        bboid &other = boids[i];
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

void limit_speed(bboid &boid)
{
    const float speed_limit = 0.01f;

    float speed = glm::sqrt(boid.dx * boid.dx + boid.dy * boid.dy);
    if (speed <= speed_limit)
        return;
    boid.dx = (boid.dx / speed) * speed_limit;
    boid.dy = (boid.dy / speed) * speed_limit;
}

void fly_towards_center(bboid &boid, std::vector<bboid> &boids)
{
    const float centering_factor = 0.005f; // adjust velocity by this %
    const float visual_range = 0.08f;

    float centerX = 0.0f;
    float centerY = 0.0f;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < boids.size(); i++)
    {
        bboid &other = boids[i];
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

float distance(bboid &boid1, bboid &boid2)
{
    return glm::sqrt(
        (boid1.x - boid2.x) * (boid1.x - boid2.x) +
            (boid1.y - boid2.y) * (boid1.y - boid2.y)
    );
}