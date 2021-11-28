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

#include <chrono>
#include <thread>

#include "boids_common.h"
#include "boids_gpu.h"
#include "boids_cpu.h"

using namespace std;

// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
const bool RUN_CPU = false;

// public
GLuint triangleVAO, triangleVBO;
const int N = 600; // 100 lub 10000
glm::vec2 *start_translations;

GLuint transformationVBO;
glm::mat4 *trans_matrices;

// set up vertex data (and buffer(s)) and configure vertex attributes
// ------------------------------------------------------------------
float vertexData[] = {
    // positions            // colors
    -0.006125f, -0.006125f, 0.0f, 0.0f, 0.0f,
    0.0f,        0.01f,     1.0f, 1.0f, 0.0f,
    0.006125f,  -0.006125f, 0.0f, 0.0f, 0.0f
};

void init_transform_resources();
void render(SDL_Window* window, Shader* shader);

// struct Boid
// {
//     float x;
//     float y;
//     float dx;
//     float dy;
// };

Boid *d_boids;
glm::mat4 *d_trans;

Shader* init_resources()
{
    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader* shader = new Shader("boids.vs", "boids.fs");
    trans_matrices = new glm::mat4[N];
    start_translations = new glm::vec2[N];
       
    for (int i = 0; i < N; i++)
    {                
        glm::vec3 translation;
        translation.x = glm::linearRand(-1.0f, 1.0f);
        translation.y = glm::linearRand(-1.0f, 1.0f);
        trans_matrices[i] = glm::translate(glm::mat4(1.0f), translation);
        start_translations[i] = translation;
    }
    
    glGenVertexArrays(1, &triangleVAO);
    glGenBuffers(1, &triangleVBO);
    glBindVertexArray(triangleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);
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
    // kazdy trojkat ma swoja macierz transformacji
    glGenBuffers(1, &transformationVBO);
    glBindBuffer(GL_ARRAY_BUFFER, transformationVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrices[0], GL_DYNAMIC_DRAW);

    // rejestracja jako zasob cuda
    // CudaGraphicsGLRegisterBuffer()
    
    glBindVertexArray(triangleVAO);
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

void print_debug(Boid &boid)
{
    cout << "boid.x = " << boid.x << endl;
    cout << "boid.y = " << boid.y << endl;
    cout << "boid.dx = " << boid.dx << endl;
    cout << "boid.dy = " << boid.dy << endl;
    return;
}

void set_initial_boid_position(Boid &boid, float *vertexData, glm::vec2 &translation)
{
    float origin_x = 0.0f;
    float origin_y = (vertexData[6] + vertexData[1]) / 2;
    boid.x = origin_x + translation.x;
    boid.y = origin_y + translation.y;    
}

Boid *init_boid_structure(int n, float *vertexData, glm::vec2 *start_translations)
{
    Boid *boids = new Boid[n];
    for (int i = 0; i < n; i++)
    {
        boids[i].dx = glm::linearRand(-0.5f, 0.5f) / 100;
        boids[i].dy = glm::linearRand(-0.5f, 0.5f) / 100;
        set_initial_boid_position(boids[i], vertexData, start_translations[i]);        
    }
    return boids;
}

void copy_boid_structure_to_device(Boid **boids, Boid **d_pointer, int n)
{
    size_t size = sizeof(Boid);
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

void copy_boid_structure_to_host(Boid **boids, Boid **d_pointer, int n)
{   
    cudaMemcpy(*boids, *d_pointer, n *  sizeof(Boid), cudaMemcpyDeviceToHost);
}

void main_loop(SDL_Window* window, Shader* shader)
{
    Boid *boids = init_boid_structure(N, vertexData, start_translations);    
    for (int i = 0; i < N; i++)
    {
        cout << i << ": " << boids[i].x << ", " << boids[i].y << endl;
    }
    copy_boid_structure_to_device(&boids, &d_boids, N);
    copy_trans_matrix_to_device(&trans_matrices, &d_trans, N);
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
        if (RUN_CPU)
        {
            cpu::cpu_test(boids, trans_matrices, N);
        }
        else
        {
            kernel_test<<<num_blocks, num_threads>>>(d_boids, d_trans, N);
            copy_trans_matrix_to_host(&trans_matrices, &d_trans, N);
        }        
        glBindBuffer(GL_ARRAY_BUFFER, transformationVBO);
        glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrices[0], GL_DYNAMIC_DRAW);
        // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
		render(window, shader);
        frame_time = SDL_GetTicks() - frame_start;

        // std::chrono::milliseconds timespan(20); // or whatever
        // std::this_thread::sleep_for(timespan);
        cout << frame_time << endl;
	}
}


void render(SDL_Window* window, Shader* shader)
{
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
    // draw N instanced triangles   
    (*shader).use();
    glBindVertexArray(triangleVAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, N); // N triangles of 3 vertices each
    glBindVertexArray(0); // zrywa binding

    SDL_GL_SwapWindow(window);
}

void free_resources(SDL_Window* window, Shader *shader)
{
    glDeleteVertexArrays(1, &triangleVAO);
    glDeleteBuffers(1, &triangleVBO);
    glDeleteBuffers(1, &transformationVBO);
    delete shader;
    delete trans_matrices;
    delete start_translations;
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

    if (window == NULL)
    {
        cerr << "Error: can't create window: " << SDL_GetError() << endl;
        return EXIT_FAILURE;
    }
    SDL_GL_CreateContext(window);

    /* Extension wrangler initialising */
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK)
    {
        cerr << "Error: glewInit: " << glewGetErrorString(glew_status) << endl;
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 1);
    // SDL_GL_SetSwapInterval(0); // wylacza vsync
    if (SDL_GL_CreateContext(window) == NULL)
    {
        cerr << "Error: SDL_GL_CreateContext: " << SDL_GetError() << endl;
        return EXIT_FAILURE;
    }

    Shader* shader = init_resources();

    main_loop(window, shader);

    free_resources(window, shader);
    return EXIT_SUCCESS;
}