#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <SDL2/SDL.h>

class Parameters
{
public:
    float centering_factor;
    float visual_range; 
    float margin;
    float turn_factor;
    float speed_limit;
    float min_distance; 
    float avoid_factor; 
    float matching_factor;
    bool avoid_mouse;

    void set_default();
    void print_values();
    bool handle_keyboard(SDL_Event &ev);
};

#endif