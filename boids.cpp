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

#include "finalBoidUtils.h"
#include <vector>
#include <chrono>
#include <thread>

using namespace std;

// settings
// const unsigned int SCR_WIDTH = 1200;
// const unsigned int SCR_HEIGHT = 900;
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
float speed_factor = 0.001f;

// public
GLuint instanceVBO;
GLuint quadVAO, quadVBO;
const int N = 500; // 100 lub 10000
glm::vec2 translations[N];

GLuint translationVBO;
glm::mat4 trans_matrix[N];

// set up vertex data (and buffer(s)) and configure vertex attributes
// ------------------------------------------------------------------
float quadVertices[] = {
    // positions     // colors
    -0.025f, -0.025f,  0.0f, 0.0f, 0.0f,
    0.0f,    0.04f,  1.0f, 1.0f, 0.0f,
    0.025f,  -0.025f,  0.0f, 0.0f, 0.0f
};

void logic();
void init_transform_resources();
void render(SDL_Window* window, Shader* shader);

Shader* init_resources()
{
    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader* shader = new Shader("boids.vs", "boids.fs");

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

void boid_logic(std::vector<bboid> &boids)
{
    for (int i = 0; i < N; i++)
    {
        bboid &boid = boids[i];
        fly_towards_center(boid, boids);
        avoid_others(boid, boids, i);
        match_velocity(boid, boids, i);
        limit_speed(boid);
        keep_within_bounds(boid, boids);
        boid.x += boid.dx;
        boid.y += boid.dy;
        // float angle = boid.dy / boid.dx;
        auto pi = glm::pi<float>();
        float angle = glm::atan(boid.dy / boid.dx);        
        // cout << "anglePrev = " << angle/pi*180 << endl;
        //float angle = glm::linearRand(-pi, pi);
        if (boid.dx <= 0)
        {
            angle += pi / 2;
        }        
        else
        {
            angle -= pi / 2;
        }
        // float angle = -pi / 4;
        // if (i == 0)
        // {
        //     print_debug(boid);
        //     cout << "dy / dx = " << boid.dy/boid.dx << endl;
        //     cout << "angle = " << angle/pi*180 << endl;            
        // }
        move_logic(boid, trans_matrix[i], angle);        
        // if (angle > pi / 2 || angle < -pi / 2)
        // {
        //     cout << angle << endl;
        //     angle = 0;
        // }
        // else
        // {
        //     angle = glm::atan(boid.dy, boid.dx);
        //     move_logic(boid, trans_matrix[i], angle);
        // }        
        // float angle = glm::atan2(boid.dy, boid.dx);
        // move_boid(boid, trans_matrix[i], boid.x, boid.y);
        // rotate_boid_and_return(boid, trans_matrix[i], angle);
    }
    glBindBuffer(GL_ARRAY_BUFFER, translationVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::mat4), &trans_matrix[0], GL_DYNAMIC_DRAW);
    // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
}

void main_loop(SDL_Window* window, Shader* shader)
{
    std::vector<bboid> boids(N);
    init_boid_structure(boids, N, quadVertices, translations);
    for (int i = 0; i < N; i++)
    {
        cout << i << ": " << boids[i].x << ", " << boids[i].y << endl;
    }
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
                //boid_logic(boids);                
            }
            if (ev.type == SDL_WINDOWEVENT &&
                ev.window.event == SDL_WINDOWEVENT_RESIZED)
            {
                glViewport(0, 0, ev.window.data1, ev.window.data2);
            }
		}
        boid_logic(boids);
        render(window, shader);
		//render(window, shader);
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