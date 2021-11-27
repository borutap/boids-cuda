#include <GL/glew.h>
#include <GLFW/glfw3.h>

/* Using SDL2 for the base window and OpenGL context init */
#include <SDL2/SDL.h>

#include "learnopengl/shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <iostream>

//#include "finalBoidUtilsKernel.h"
#include <vector>
#include <chrono>
#include <thread>

using namespace std;

// settings
// const unsigned int SCR_WIDTH = 1200;
// const unsigned int SCR_HEIGHT = 900;
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
float speed_factor = 0.001f;

// public
GLuint instanceVBO;
GLuint quadVAO, quadVBO;
const int N = 8000; // 100 lub 10000
glm::vec2 *translations;

GLuint translationVBO;
glm::mat4 *trans_matrix;

// set up vertex data (and buffer(s)) and configure vertex attributes
// ------------------------------------------------------------------
float quadVertices[] = {
    // positions     // colors
    -0.006125f, -0.006125f,  0.0f, 0.0f, 0.0f,
    0.0f,    0.01f,  1.0f, 1.0f, 0.0f,
    0.006125f,  -0.006125f,  0.0f, 0.0f, 0.0f
};

void logic();
void init_transform_resources();
void render(SDL_Window* window, Shader* shader);

struct bboid
{
    float x;
    float y;
    float dx;
    float dy;
};

bboid *d_boids;
glm::mat4 *d_trans;

Shader* init_resources()
{
    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader* shader = new Shader("boids.vs", "boids.fs");
    trans_matrix = new glm::mat4[N];
    translations = new glm::vec2[N];
    // tu zmienic przy zmianie N na 10000    
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {                
        glm::vec3 translation;
        translation.x = glm::linearRand(-1.0f, 1.0f);
        translation.y = glm::linearRand(-1.0f, 1.0f);
        trans_matrix[i] = glm::translate(glm::mat4(1.0f), translation);
        translations[i] = translation;
    }
    // for (int y = -10; y < 10; y += 2)
    // {
    //     for (int x = -10; x < 10; x += 2)
    //     {
    //         glm::vec2 translation;
    //         translation.x = (float)x / 10.0f + offset;
    //         translation.y = (float)y / 10.0f + offset;
    //         translations[index++] = translation;
    //     }
    // }

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    // wspolne dane
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,                  // nr atrybutu - zgodny z layoutem w .vs
        2,                  // rozmiar
        GL_FLOAT,           // typ
        GL_FALSE,           // czy znormalizowane
        5 * sizeof(float),  // stride - odstep do kolejnej wartosci
        (void*)0            // offset jezeli jest we wspolnej tablicy
    );
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    
    init_transform_resources();

    return shader;
}

void init_transform_resources()
{    
    glGenBuffers(1, &translationVBO);
    glBindBuffer(GL_ARRAY_BUFFER, translationVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrix[0], GL_DYNAMIC_DRAW);
    
    glBindVertexArray(quadVAO);
    // set attribute pointers for matrix (4 times vec4)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
    glVertexAttribDivisor(6, 1);

    glBindVertexArray(0);
}




void print_debug(bboid &boid)
{
    cout << "boid.x = " << boid.x << endl;
    cout << "boid.y = " << boid.y << endl;
    cout << "boid.dx = " << boid.dx << endl;
    cout << "boid.dy = " << boid.dy << endl;
    return;
}

// void boid_logic(std::vector<bboid> &boids)
// {
//     for (int i = 0; i < N; i++)
//     {
//         bboid &boid = boids[i];
//         fly_towards_center(boid, boids);
//         avoid_others(boid, boids, i);
//         match_velocity(boid, boids, i);
//         limit_speed(boid);
//         keep_within_bounds(boid, boids);
//         boid.x += boid.dx;
//         boid.y += boid.dy;
//         // float angle = boid.dy / boid.dx;
//         auto pi = glm::pi<float>();
//         float angle = glm::atan(boid.dy / boid.dx);        
//         // cout << "anglePrev = " << angle/pi*180 << endl;
//         //float angle = glm::linearRand(-pi, pi);
//         if (boid.dx <= 0)
//         {
//             angle += pi / 2;
//         }        
//         else
//         {
//             angle -= pi / 2;
//         }
//         // float angle = -pi / 4;
//         // if (i == 0)
//         // {
//         //     print_debug(boid);
//         //     cout << "dy / dx = " << boid.dy/boid.dx << endl;
//         //     cout << "angle = " << angle/pi*180 << endl;            
//         // }
//         move_logic(boid, trans_matrix[i], angle);        
//         // if (angle > pi / 2 || angle < -pi / 2)
//         // {
//         //     cout << angle << endl;
//         //     angle = 0;
//         // }
//         // else
//         // {
//         //     angle = glm::atan(boid.dy, boid.dx);
//         //     move_logic(boid, trans_matrix[i], angle);
//         // }        
//         // float angle = glm::atan2(boid.dy, boid.dx);
//         // move_boid(boid, trans_matrix[i], boid.x, boid.y);
//         // rotate_boid_and_return(boid, trans_matrix[i], angle);
//     }
//     glBindBuffer(GL_ARRAY_BUFFER, translationVBO);
//     glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrix[0], GL_DYNAMIC_DRAW);
//     // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
//     glBindBuffer(GL_ARRAY_BUFFER, 0); 
// }

void set_initial_boid_position(bboid &boid, float *quadVertices, glm::vec2 &translation)
{
    float origin_x = 0.0f;
    float origin_y = (quadVertices[6] + quadVertices[1]) / 2;
    boid.x = origin_x + translation.x;
    boid.y = origin_y + translation.y;    
}

void init_boid_structure(bboid *boids, int n, 
                         float *quadVertices, glm::vec2 *translations)
{
    for (int i = 0; i < n; i++)
    {
        boids[i].dx = glm::linearRand(-0.5f, 0.5f) / 100;
        boids[i].dy = glm::linearRand(-0.5f, 0.5f) / 100;
        set_initial_boid_position(boids[i], quadVertices, translations[i]);        
    }
}

__device__ float distance(bboid &boid1, bboid &boid2)
{
    return glm::sqrt(
        (boid1.x - boid2.x) * (boid1.x - boid2.x) +
            (boid1.y - boid2.y) * (boid1.y - boid2.y)
    );
}

__device__ void fly_towards_center(bboid *boids, int index, int n)
{    
    bboid &boid = boids[index];
    const float centering_factor = 0.005f; // adjust velocity by this %
    const float visual_range = 0.07f;

    float centerX = 0.0f;
    float centerY = 0.0f;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < n; i++)
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

__device__ void keep_within_bounds(bboid *boids, int index)
{
    bboid &boid = boids[index];
    const float margin = 0.05f;
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

__device__ void limit_speed(bboid *boids, int index)
{
    bboid &boid = boids[index];
    const float speed_limit = 0.01f;

    float speed = glm::sqrt(boid.dx * boid.dx + boid.dy * boid.dy);
    if (speed <= speed_limit)
        return;
    boid.dx = (boid.dx / speed) * speed_limit;
    boid.dy = (boid.dy / speed) * speed_limit;
}

__device__ void avoid_others(bboid *boids, int index, int n)
{
    bboid &boid = boids[index];
    const float min_distance = 0.01f; // The distance to stay away from other boids
    const float avoid_factor = 0.05f; // Adjust velocity by this %
    float moveX = 0;
    float moveY = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        if (i == index)
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

__device__ void match_velocity(bboid *boids, int index, int n)
{
    bboid &boid = boids[index];
    const float matching_factor = 0.05f;
    const float visual_range = 0.07f; // TODO - to do glob. zmiennej

    float avgDX = 0.0f;
    float avgDY = 0.0f;
    int num_neighbors = 0;

    for (unsigned int i = 0; i < n; i++)
    {
        if (i == index)
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

__global__ void kernel_test(bboid *boids, glm::mat4 *trans, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;
    fly_towards_center(boids, index, n);
    avoid_others(boids, index, n);
    match_velocity(boids, index, n);
    limit_speed(boids, index);
    keep_within_bounds(boids, index);
    boids[index].x += boids[index].dx;
    boids[index].y += boids[index].dy;
    float angle = glm::atan(boids[index].dy / boids[index].dx);
    float pi = glm::pi<float>();
    if (boids[index].dx <= 0)
    {
        angle += pi / 2;
    }        
    else
    {
        angle -= pi / 2;
    }
    auto rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f));
    auto move2 = glm::translate(glm::mat4(1.0f), glm::vec3(boids[index].x, boids[index].y, 0.0f));
    trans[index] = move2 * rotation;
    // printf("boids[%d].x = %f\n", index, boid.x);
    // cout << "boid.x = " << boid.x << endl;
    // cout << "boid.y = " << boid.y << endl;
    // cout << "boid.dx = " << boid.dx << endl;
    // cout << "boid.dy = " << boid.dy << endl;
}



void copy_boid_structure_to_device(bboid **boids, bboid **d_pointer, int n)
{
    size_t size = sizeof(bboid);
    cudaMalloc(d_pointer, n * size);
    cudaMemcpy(*d_pointer, *boids, n * size, cudaMemcpyHostToDevice);
}

void copy_trans_matrix_to_device(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    size_t size = sizeof(glm::mat4);
    cudaMalloc(d_mat, n * size);
    cudaMemcpy(*d_mat, *mat, n * size, cudaMemcpyHostToDevice);
}

void copy_trans_matrix_to_host(glm::mat4 **mat, glm::mat4 **d_mat, int n)
{
    cudaMemcpy(*mat, *d_mat, n *  sizeof(glm::mat4), cudaMemcpyDeviceToHost);
}

void copy_boid_structure_to_host(bboid **boids, bboid **d_pointer, int n)
{   
    cudaMemcpy(*boids, *d_pointer, n *  sizeof(bboid), cudaMemcpyDeviceToHost);
}

void main_loop(SDL_Window* window, Shader* shader)
{
    bboid *boids = new bboid[N];
    init_boid_structure(boids, N, quadVertices, translations);
    for (int i = 0; i < N; i++)
    {
        cout << i << ": " << boids[i].x << ", " << boids[i].y << endl;
    }
    copy_boid_structure_to_device(&boids, &d_boids, N);
    copy_trans_matrix_to_device(&trans_matrix, &d_trans, N);
    dim3 num_threads(1024);
    dim3 num_blocks(N / 1024 + 1);
    while (true) {
        Uint32 frame_start = SDL_GetTicks();
        int frame_time;
		SDL_Event ev;
		while (SDL_PollEvent(&ev)) {
			if (ev.type == SDL_QUIT)
				return;
            if (ev.type == SDL_KEYDOWN &&
                ev.key.keysym.sym == SDLK_r)
            {
                // kernel_test<<<num_blocks, num_threads>>>(d_boids, N); 
                // cudaDeviceSynchronize();
                // copy_boid_structure_to_host(&boids, &d_boids, N);
                // for (int i = 0; i < N; i++)
                // {
                //     cout << "yes " << i << ": " << boids[i].x << ", " << boids[i].y << endl;
                // }
            }
            if (ev.type == SDL_WINDOWEVENT &&
                ev.window.event == SDL_WINDOWEVENT_RESIZED)
            {
                glViewport(0, 0, ev.window.data1, ev.window.data2);
            }
		}
        //boid_logic(boids);
        //render(window, shader);
        kernel_test<<<num_blocks, num_threads>>>(d_boids, d_trans, N);
        //cudaDeviceSynchronize();
        copy_trans_matrix_to_host(&trans_matrix, &d_trans, N);
        glBindBuffer(GL_ARRAY_BUFFER, translationVBO);
        glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrix[0], GL_DYNAMIC_DRAW);
        // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
		render(window, shader);
        frame_time = SDL_GetTicks() - frame_start;
        // for (int i = 0; i < N; i++)
        // {
        //     cout << i << ": " << boids[i].pos_x << ", " << boids[i].pos_y << endl;
        // }
        
        // std::chrono::milliseconds timespan(20); // or whatever

        // std::this_thread::sleep_for(timespan);
        cout << frame_time << endl;
	}
}

void logic()
{
    float move = sinf(SDL_GetTicks() / 10000.0 * (2*3.14) / 5); 
    // // 45° per second
	// // float angle = SDL_GetTicks() / 1000.0 * 45;
	// // glm::vec3 axis_z(0, 1, 0);
	// //glm::mat4 m_transform = glm::translate(glm::mat4(1.0f), glm::vec3(move, 0.0, 0.0));
    int index = SDL_GetTicks() % 100;
    if (rand() % 2 == 0)
        move = -move;
    translations[index].x = translations[index].x + move;
    translations[index].y = translations[index].y + move/2;
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
    //glBufferSubData
    //cout << translations[SDL_GetTicks() % 100].x << endl;
}

void render(SDL_Window* window, Shader* shader)
{
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
    // draw 100 instanced quads   
    (*shader).use();
    glBindVertexArray(quadVAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, N); // 100 triangles of 3 vertices each
    glBindVertexArray(0); // zrywa binding

    SDL_GL_SwapWindow(window);
}

void free_resources(SDL_Window* window, Shader *shader)
{
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteBuffers(1, &translationVBO);
    free(shader);
    free(trans_matrix);
    free(translations);
    cudaFree(d_trans);
    cudaFree(d_boids);
    SDL_DestroyWindow(window);

    //Quit SDL subsystems
    SDL_Quit();
    cout << "Wyczyszczono" << endl;
}

int main()
{
    /* SDL-related initialising functions */
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* window = SDL_CreateWindow("Boids",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		SCR_WIDTH, SCR_HEIGHT,
		SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);

    if (window == NULL) {
        cerr << "Error: can't create window: " << SDL_GetError() << endl;
        return EXIT_FAILURE;
    }
	SDL_GL_CreateContext(window);

	/* Extension wrangler initialising */
	GLenum glew_status = glewInit();
	if (glew_status != GLEW_OK) {
		cerr << "Error: glewInit: " << glewGetErrorString(glew_status) << endl;
		return EXIT_FAILURE;
	}

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 1);
    // SDL_GL_SetSwapInterval(0); // wylacza vsync
	if (SDL_GL_CreateContext(window) == NULL) {
		cerr << "Error: SDL_GL_CreateContext: " << SDL_GetError() << endl;
		return EXIT_FAILURE;
	}

    Shader* shader = init_resources();

    main_loop(window, shader);

    free_resources(window, shader);
    return EXIT_SUCCESS;
}