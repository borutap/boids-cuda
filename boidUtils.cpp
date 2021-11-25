#include "boidUtils.h"

void myMain(std::vector<boid> &boids, int n)
{
    //return;
    float dist = 0.0f;
    float deg = 0.0f;

    for (int i = 0; i < n; i++)
    {
        boids[i].neighbors_mean_speed_x = boids[i].speed_x;
        boids[i].neighbors_mean_speed_y = boids[i].speed_y;
        boids[i].neighbors_mean_distance = 0;
        boids[i].neighbor_count = 1;
        for (int j = 0; j < n; j++)
        {
            if (j == i) continue;
            dist = glm::sqrt(
                glm::pow(boids[i].pos_x - boids[j].pos_x, 2) + glm::pow(
                    boids[i].pos_y - boids[j].pos_y, 2)
            );
            deg = glm::acos(
                (boids[i].speed_x /
                glm::sqrt(boids[i].speed_x * boids[i].speed_x + boids[i].speed_y * boids[i].speed_y)) *
                ((boids[j].pos_x - boids[i].pos_x) / dist) +
                (boids[i].speed_y /
                glm::sqrt(boids[i].speed_x * boids[i].speed_x + boids[i].speed_y * boids[i].speed_y)) *
                ((boids[j].pos_y - boids[i].pos_y) / dist)
            );
            deg = glm::abs((180 * deg) / glm::pi<float>());
            if (dist < MAX_DISTANCE && deg < VIEW_DEGREE)
            {
                boids[i].neighbor_count++;
                boids[i].neighbors_mean_speed_x += boids[j].speed_x;
                boids[i].neighbors_mean_speed_y += boids[j].speed_y;
                boids[i].neighbors_mean_distance += dist;
            }
        }
    }


    for (int i = 0; i < n; i++)
    {
        //adjust speed to neighbours speed
        boids[i].speed_x += WEIGHT_SPEED * (boids[i].neighbors_mean_speed_x / boids[i].neighbor_count - boids[i].speed_x);
        boids[i].speed_y += WEIGHT_SPEED * (boids[i].neighbors_mean_speed_y / boids[i].neighbor_count - boids[i].speed_y);

        //pertubation
        float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        boids[i].speed_x += WEIGHT_RANDOM * ((random - 0.5) * MAX_SPEED);
        boids[i].speed_y += WEIGHT_RANDOM * ((random - 0.5) * MAX_SPEED);

        if (boids[i].neighbor_count > 1) boids[i].neighbors_mean_distance /= boids[i].neighbor_count - 1;
        for (int j = 0; j < n; j++)
        {
            if (j == i) continue;
            dist = glm::sqrt(
                glm::pow(boids[i].pos_x - boids[j].pos_x, 2) + glm::pow(boids[i].pos_y - boids[j].pos_y, 2)
            );
            deg = glm::acos(
                (boids[i].speed_x /
                glm::sqrt(boids[i].speed_x * boids[i].speed_x + boids[i].speed_y * boids[i].speed_y)) *
                ((boids[j].pos_x - boids[i].pos_x) / dist) +
                (boids[i].speed_y /
                glm::sqrt(boids[i].speed_x * boids[i].speed_x + boids[i].speed_y * boids[i].speed_y)) *
                ((boids[j].pos_y - boids[i].pos_y) / dist)
            );
            deg = glm::abs((180 * deg) / glm::pi<float>());
            if (dist < MAX_DISTANCE && deg < VIEW_DEGREE)
            {
                if (glm::abs(boids[j].pos_x - boids[i].pos_x) > D_MIN)
                {
                    boids[i].speed_x +=
                        (WEIGHT_DISTANCE / boids[i].neighbor_count) *
                        (((boids[j].pos_x - boids[i].pos_x) * (dist - boids[i].neighbors_mean_distance)) / dist);
                    boids[i].speed_y +=
                    (WEIGHT_DISTANCE / boids[i].neighbor_count) *
                    (((boids[j].pos_y - boids[i].pos_y) * (dist - boids[i].neighbors_mean_distance)) / dist);
                }
            
            } //neighbours are too close
            else
            {
                boids[i].speed_x -=
                    (WEIGHT_MIN / boids[i].neighbor_count) *
                    (((boids[j].pos_x - boids[i].pos_x) * D_MIN) / dist -
                    (boids[j].pos_x - boids[i].pos_x));
                boids[i].speed_y -=
                (WEIGHT_MIN / boids[i].neighbor_count) *
                (((boids[j].pos_y - boids[i].pos_y) * D_MIN) / dist -
                (boids[j].pos_y - boids[i].pos_y));            
            }
        }

        //check speed is not too high
        if (glm::sqrt(boids[i].speed_x * boids[i].speed_x + boids[i].speed_y * boids[i].speed_y) > MAX_SPEED)
        {
            boids[i].speed_x *= 0.25;
            boids[i].speed_y *= 0.25;
        }
    } 
}

glm::vec2 convertToBig(glm::vec2 &original, int scr_width, int scr_height)
{
    glm::vec2 ret = glm::vec2(original.x + 1, original.y + 1);
    ret.x *= scr_width/2;
    ret.y *= scr_height/2;
    return ret;
}

void convertToSmall(boid &boid, int scr_width, int scr_height)
{
    boid.pos_x /= scr_width/2;
    boid.pos_x -= 1;
    boid.pos_y /= scr_height/2;
    boid.pos_y -= 1;
}

void convertToBig(boid &boid, int scr_width, int scr_height)
{
    boid.pos_x += 1;
    boid.pos_x *= scr_width/2;    
    boid.pos_y += 1;
    boid.pos_y *= scr_height/2;    
}

void init_boids(std::vector<boid> &boids, float *quadVertices, glm::vec2 *translations, int n,
                int scr_width, int scr_height)
{    
    for (int i = 0; i < n; i++)
    {
        boids[i].speed_x = SPEED_FACTOR_X;
        boids[i].speed_y = SPEED_FACTOR_Y;
        auto pos = get_initial_boid_position(quadVertices, translations[i]);
        pos = convertToBig(pos, scr_width, scr_height);
        boids[i].pos_x = pos.x;
        boids[i].pos_y = pos.y;
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
        thisB.speed_x += WEIGHT_SPEED * (thisB.neighbors_mean_speed_x - thisB.speed_x);

        auto thisPos = glm::vec2(thisB.pos_x, thisB.pos_y);
        // losowa zmiana predkosci
        // random z [0,1]
        float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        thisB.speed_x += WEIGHT_RANDOM * ((random - 0.5f) * MAX_SPEED);         
        for (int j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            auto neighborPos = glm::vec2(boids[j].pos_x, boids[j].pos_y);
            float distance = boid_distance_to_neighbor(neighborPos, thisPos);
            float angle = move_angle(thisB.speed_y, thisB.speed_x);
            float neighbor_angle = boid_angle_to_neighbor(neighborPos, thisPos);
            if (is_neighbor_distance(distance, MAX_DISTANCE) &&
                is_neighbor_angle(angle, neighbor_angle, VIEW_DEGREE))
            {
                if (glm::abs(boids[j].pos_x - thisB.pos_x) > D_MIN)
                {
                    thisB.speed_x += (WEIGHT_DISTANCE / thisB.neighbor_count) *
                                        (((boids[j].pos_x - thisB.pos_x) * 
                                            (distance - thisB.neighbors_mean_distance)) /
                                        distance);
                    thisB.speed_y += (WEIGHT_DISTANCE / thisB.neighbor_count) *
                                        (((boids[j].pos_y - thisB.pos_y) * 
                                            (distance - thisB.neighbors_mean_distance)) /
                                        distance);
                }
                
            } // sasiedzi sa za blisko
            // else if (glm::all(glm::isnan(thisB.speed)))
            // {
            //     std::cout << "errorrrrrrrrrrrrrrrrr" << std::endl;
            // }
            else
            {
                // if (glm::abs(distance - 0.001f) < 0.01f)
                //     continue;
                thisB.speed_x -= (WEIGHT_MIN / thisB.neighbor_count) *
                    (((boids[j].pos_x - thisB.pos_x) * D_MIN) / distance -
                    (boids[j].pos_x - thisB.pos_x));
                thisB.speed_y -= (WEIGHT_MIN / thisB.neighbor_count) *
                    (((boids[j].pos_y - thisB.pos_y) * D_MIN) / distance -
                    (boids[j].pos_y - thisB.pos_y));
                // if (glm::all(glm::isnan(thisB.speed)))
                // {
                //     std::cout << "errorrrrr" << std::endl;
                // }
            }
        }
        // czy nie za szybko
        if (glm::sqrt(thisB.speed_x * thisB.speed_x + thisB.speed_y * thisB.speed_y) > MAX_SPEED)
        {
            thisB.speed_x *= 0.15f;        
            thisB.speed_y *= 0.15f;        
        }
            
    }
}

void neighbor_search_action(std::vector<boid> &boids, int n, float *quadVertices, glm::vec2 *translations)
{
    for (int i = 0; i < n; i++)
    {        
        boids[i].neighbors_mean_speed_x = boids[i].speed_x;
        boids[i].neighbors_mean_speed_y = boids[i].speed_y;
        boids[i].neighbors_mean_distance = 0;
        boids[i].neighbor_count = 1; // wliczamy siebie
        auto thisPos = glm::vec2(boids[i].pos_x, boids[i].pos_y);
        for (int j = 0; j < n; j++)
        {
            if (j == i)
                continue;
            auto neighborPos = glm::vec2(boids[j].pos_x, boids[j].pos_y);
            float distance = boid_distance_to_neighbor(neighborPos, thisPos);
            if (!is_neighbor_distance(distance, MAX_DISTANCE))
                continue;

            float angle = move_angle(boids[i].speed_y, boids[i].speed_x);
            float neighbor_angle = boid_angle_to_neighbor(neighborPos, thisPos);
            if (!is_neighbor_angle(angle, neighbor_angle, VIEW_DEGREE))
                continue;

            // jest sasiadem
            boids[i].neighbor_count++;
            boids[i].neighbors_mean_speed_x += boids[j].speed_x;
            boids[i].neighbors_mean_speed_y += boids[j].speed_y;
            boids[i].neighbors_mean_distance += distance;
        }
        if (boids[i].neighbor_count == 1) // brak sasiadow
            return;
        boids[i].neighbors_mean_speed_x /= boids[i].neighbor_count;
        boids[i].neighbors_mean_speed_y /= boids[i].neighbor_count;
        boids[i].neighbors_mean_distance /= (boids[i].neighbor_count - 1);
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
                    boid &boid)
{
    float angle;
    float PI = glm::pi<float>();
    float speed_angle = glm::atan(boid.speed_y / boid.speed_x);
    if (boid.speed_x < 0.0f)
    {
        angle = PI / 4.0f + speed_angle; // 90 stopni + kat z predkosci
    }
    else
    {
        angle = -PI / 4.0f + speed_angle; // -90 stopni + kat z predkosci
    }
    move_boid(matrix, origin_translation, boid.pos_x, boid.pos_y);
    // glm::mat4 translation1 = glm::translate(glm::mat4(1.0f), glm::vec3(-origin_translation.x, -origin_translation.y, 0.0));
    // // przeniesienie
    // // TODO
    // move_translation = glm::translate(move_translation, glm::vec3(pos.x, pos.y, 0.0));
    // // rotacja
    rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0, 0.0, 1.0));
    matrix = matrix * rotation;
    // // powrot na miejsce
    // glm::mat4 translation2 = glm::translate(glm::mat4(1.0f), glm::vec3(origin_translation.x, origin_translation.y, 0.0));

    // matrix = translation2 * rotation * move_translation * translation1;    
}

void move_boid(glm::mat4 &matrix, glm::vec2 &origin_translation, float x, float y)
{    
    glm::mat4 translation = glm::translate(
        glm::mat4(1.0f), glm::vec3(-origin_translation.x + x, -origin_translation.y + y, 0.0));
    matrix = translation;
}