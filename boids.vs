#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aOffset; // zamiast tego moze macierz transformacji
layout (location = 3) in mat4 aTransform; // zamiast tego moze macierz transformacji

out vec3 fColor;

void main()
{
    fColor = aColor;
    gl_Position = aTransform * vec4(aPos + aOffset, 0.0, 1.0);
    //gl_Position = vec4(aPos + aOffset, 0.0, 1.0);
}