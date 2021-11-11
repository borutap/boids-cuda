#include <GL/glew.h>
#include <GLFW/glfw3.h>

/* Using SDL2 for the base window and OpenGL context init */
#include <SDL2/SDL.h>

#include "learnopengl/shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
using namespace std;

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// public
GLuint instanceVBO;
GLuint quadVAO, quadVBO;
glm::vec2 translations[100];

GLuint rotationVBO;
glm::mat4 rot_matrix[100];

void logic();
void rotation_resources();
void render(SDL_Window* window, Shader* shader);

Shader* init_resources()
{
    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader* shader = new Shader("boids.vs", "boids.fs");
    
    // generate a list of 100 quad locations/translation-vectors
    // ---------------------------------------------------------
    //glm::vec2 translations[100];
    int index = 0;
    float offset = 0.1f;
    for (int y = -10; y < 10; y += 2)
    {
        for (int x = -10; x < 10; x += 2)
        {
            glm::vec2 translation;
            translation.x = (float)x / 10.0f + offset;
            translation.y = (float)y / 10.0f + offset;
            translations[index++] = translation;
        }
    }

    // store instance data in an array buffer
    // --------------------------------------
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float quadVertices[] = {
        // positions     // colors
        -0.05f, -0.05f,  0.0f, 0.0f, 0.0f,
        0.0f,    0.08f,  1.0f, 1.0f, 0.0f,
        0.05f,  -0.05f,  0.0f, 0.0f, 0.0f
    };
    
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
    // also set instance data
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO); // this attribute comes from a different vertex buffer
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(2, 1); // tell OpenGL this is an instanced vertex attribute.

    rotation_resources();

    return shader;
}

void rotation_resources()
{
    for (int i = 0; i < 100; i++)
        rot_matrix[i] = glm::mat4(1.0f);

    glGenBuffers(1, &rotationVBO);
    glBindBuffer(GL_ARRAY_BUFFER, rotationVBO);
    glBufferData(GL_ARRAY_BUFFER, 100 * sizeof(glm::mat4), &rot_matrix[0], GL_STATIC_DRAW);
    
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

void logic_rotate()
{
    static int index = 0;
    // translations[index].x = 0;
    // translations[index].y = 0;
    // glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    // //glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
    // glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * index, sizeof(glm::vec2), &translations[index]);
    glm::mat4& matrix = rot_matrix[index];
    //matrix = glm::rotate(matrix, glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
    matrix = glm::translate(matrix, glm::vec3(0.3, 0.3, 0.0));
    glBindBuffer(GL_ARRAY_BUFFER, rotationVBO);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * index, sizeof(glm::mat4), &matrix);
    cout << index << endl;
    index++;
    if (index == 100)
        index = 0;
}

void main_loop(SDL_Window* window, Shader* shader)
{
    while (true) {
		SDL_Event ev;
		while (SDL_PollEvent(&ev)) {
			if (ev.type == SDL_QUIT)
				return;
            if (ev.type == SDL_KEYDOWN &&
                ev.key.keysym.sym == SDLK_r)
            {
                logic_rotate();
            }
		}
        //logic();
		render(window, shader);
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
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 100); // 100 triangles of 3 vertices each
    glBindVertexArray(0); // zrywa binding

    SDL_GL_SwapWindow(window);
}

void free_resources(SDL_Window* window, Shader *shader)
{
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &instanceVBO);
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
	if (SDL_GL_CreateContext(window) == NULL) {
		cerr << "Error: SDL_GL_CreateContext: " << SDL_GetError() << endl;
		return EXIT_FAILURE;
	}

    Shader* shader = init_resources();

    main_loop(window, shader);

    free_resources(window, shader);
    return EXIT_SUCCESS;
}