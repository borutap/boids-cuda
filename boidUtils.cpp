#include "boidUtils.h"

void init_boids(std::vector<boid> &boids, float *quadVertices, glm::vec2 *translations, int n)
{    
    for (int i = 0; i < n; i++)
    {
        boids[i].speed = glm::vec2(SPEED_FACTOR_X, SPEED_FACTOR_Y);
        boids[i].pos = get_initial_boid_position(quadVertices, translations[i]);
        //boids[i].neighbors_mean_speed = glm::vec2(0.0f, 0.0f);
        // if (glm::all(glm::isnan(boids[i].pos)))
        // {
        //     cout << "error" << endl;
        // }
    }
    srand(time(NULL));
}

void on_update(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations)
{     
    neighbor_search_action(boids, n, quadVertices, translations);
    for (int i = 0; i < n; i++)
    {
        boid &thisB = boids[i];
        // dostosowanie predkosci do predkosci sasiadow
        thisB.speed += WEIGHT_SPEED * (thisB.neighbors_mean_speed - thisB.speed);

        // losowa zmiana predkosci
        // random z [0,1]
        float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        thisB.speed += WEIGHT_RANDOM * ((random - 0.5f) * MAX_SPEED);         
        for (int j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            float distance = boid_distance_to_neighbor(boids[j].pos, thisB.pos);
            float angle = move_angle(thisB.speed.y, thisB.speed.x);
            float neighbor_angle = boid_angle_to_neighbor(boids[j].pos, thisB.pos);
            if (is_neighbor_distance(distance, MAX_DISTANCE) &&
                is_neighbor_angle(angle, neighbor_angle, VIEW_DEGREE))
            {
                if (glm::abs(boids[j].pos.x - thisB.pos.x) > D_MIN)
                {
                    thisB.speed.x += (WEIGHT_DISTANCE / thisB.neighbor_count) *
                                        (((boids[j].pos.x - thisB.pos.x) * 
                                            (distance - thisB.neighbors_mean_distance)) /
                                        distance);
                    thisB.speed.y += (WEIGHT_DISTANCE / thisB.neighbor_count) *
                                        (((boids[j].pos.y - thisB.pos.y) * 
                                            (distance - thisB.neighbors_mean_distance)) /
                                        distance);
                }
                
            } // sasiedzi sa za blisko
            else if (glm::all(glm::isnan(thisB.speed)))
            {
                std::cout << "errorrrrrrrrrrrrrrrrr" << std::endl;
            }
            else
            {
                if (glm::abs(distance - 0.001f) < 0.01f)
                    continue;
                thisB.speed.x -= (WEIGHT_MIN / thisB.neighbor_count) *
                    (((boids[j].pos.x - thisB.pos.x) * D_MIN) / distance -
                    (boids[j].pos.x - thisB.pos.x));
                thisB.speed.y -= (WEIGHT_MIN / thisB.neighbor_count) *
                    (((boids[j].pos.y - thisB.pos.y) * D_MIN) / distance -
                    (boids[j].pos.y - thisB.pos.y));
                if (glm::all(glm::isnan(thisB.speed)))
                {
                    std::cout << "errorrrrr" << std::endl;
                }
            }
        }
        // czy nie za szybko
        if (glm::length(thisB.speed) > MAX_SPEED)
            thisB.speed *= 0.75f;        
    }
}

void neighbor_search_action(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations)
{
    for (int i = 0; i < n; i++)
    {        
        boids[i].neighbors_mean_speed = glm::vec2(boids[i].speed.x, boids[i].speed.y);
        boids[i].neighbors_mean_distance = 0;
        boids[i].neighbor_count = 1; // wliczamy siebie
        for (int j = 0; j < n; j++)
        {
            if (j == i)
                continue;

            float distance = boid_distance_to_neighbor(boids[j].pos, boids[i].pos);
            if (!is_neighbor_distance(distance, MAX_DISTANCE))
                continue;

            float angle = move_angle(boids[i].speed.y, boids[i].speed.x);
            float neighbor_angle = boid_angle_to_neighbor(boids[j].pos, boids[i].pos);
            if (!is_neighbor_angle(angle, neighbor_angle, VIEW_DEGREE))
                continue;

            // jest sasiadem
            boids[i].neighbor_count++;
            boids[i].neighbors_mean_speed += boids[j].speed;
            boids[i].neighbors_mean_distance += distance;
        }
        if (boids[i].neighbor_count == 1) // brak sasiadow
            return;
        boids[i].neighbors_mean_speed /= boids[i].neighbor_count;
        boids[i].neighbors_mean_distance /= boids[i].neighbor_count - 1;
    }
}

glm::vec2 get_initial_boid_position(float *quadVertices, glm::vec2 &translation)
{
    // gets center of triangle
    float origin_x = 0.0f;
    float origin_y = (quadVertices[6] + quadVertices[1]) / 2;
    return glm::vec2(origin_x, origin_y) + translation;    
}

// matrix to wyjsciowa macierz transformacji
void transform_boid(glm::mat4 &matrix, glm::mat4 &move_translation,
                    glm::mat4 &rotation, glm::vec2 &origin_translation,
                    glm::vec2 &speed, glm::vec2 &pos)
{
    float angle;
    float PI = glm::pi<float>();
    float speed_angle = glm::atan(speed.y / speed.x);
    if (speed.x < 0.0f)
    {
        angle = PI / 4.0f + speed_angle; // 90 stopni + kat z predkosci
    }
    else
    {
        angle = -PI / 4.0f + speed_angle; // -90 stopni + kat z predkosci
    }
    move_boid(matrix, origin_translation, pos.x, pos.y);
    // glm::mat4 translation1 = glm::translate(glm::mat4(1.0f), glm::vec3(-origin_translation.x, -origin_translation.y, 0.0));
    // // przeniesienie
    // // TODO
    // move_translation = glm::translate(move_translation, glm::vec3(pos.x, pos.y, 0.0));
    // // rotacja
    rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0, 0.0, 1.0));
    matrix = rotation * matrix;
    // // powrot na miejsce
    // glm::mat4 translation2 = glm::translate(glm::mat4(1.0f), glm::vec3(origin_translation.x, origin_translation.y, 0.0));

    // matrix = translation2 * rotation * move_translation * translation1;    
}

void move_boid(glm::mat4 &matrix, glm::vec2 &origin_translation, float x, float y)
{    
    glm::mat4 translation = glm::translate(
        glm::mat4(1.0f), glm::vec3(-origin_translation.x + x, -origin_translation.y, 0.0));
    matrix = translation;
}