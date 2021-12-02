#include "parameters.h"
#include <iostream>

using namespace std;

void Parameters::set_default()
{
    centering_factor = 0.002f;
    visual_range = 0.05f;
    margin = 0.1f;
    turn_factor = 0.0005f;
    speed_limit = 0.005f;
    min_distance = 0.014f;
    avoid_factor = 0.05f; 
    matching_factor = 0.05f;
    avoid_mouse = false;
}

void Parameters::print_values()
{
    cout << endl;
    cout << "(Fly towards center)                 Centering factor = " << centering_factor
        << " (-q +w)" << endl;
    cout << "(Fly towards center, Match Velocity) Visual range     = " << visual_range
        << " (-a +s)" << endl;
    cout << "(Keep within bounds) Margin                           = " << margin
        << " (-z +x)" << endl;
    cout << "(Keep within bounds) Turn factor                      = " << turn_factor
        << " (-e +r)" << endl;
    cout << "                     Speed limit                      = " << speed_limit
        << " (-d +f)" << endl;
    cout << "(Avoid others)       Min distance                     = " << min_distance
        << " (-c +v)" << endl;
    cout << "(Avoid others)       Avoid factor                     = " << avoid_factor
        << " (-t +y)" << endl;
    cout << "(Match velocity)     Matching factor                  = " << matching_factor
        << " (-g +h)" << endl;
    cout << "Set defaults (p)   Avoid mouse (m): ";
    if (avoid_mouse)
        cout << "YES";
    else
        cout << "NO";
    cout << endl;
}

bool Parameters::handle_keyboard(SDL_Event &ev)
{
    switch (ev.key.keysym.sym)
    {
        case SDLK_q:
            centering_factor -= 0.0005f;
        break;

        case SDLK_w:
            centering_factor += 0.0005f;
        break;

        case SDLK_a:
            visual_range -= 0.01f;
        break;

        case SDLK_s:
            visual_range += 0.01f;
        break;

        case SDLK_z:
            margin -= 0.01f;
        break;

        case SDLK_x:
            margin += 0.01f;
        break;

        case SDLK_e:
            turn_factor -= 0.0001f;
        break;

        case SDLK_r:
            turn_factor += 0.0001f;
        break;

        case SDLK_d:
            speed_limit -= 0.001f;
        break;

        case SDLK_f:
            speed_limit += 0.001f;
        break;

        case SDLK_c:
            min_distance -= 0.001f;
        break;

        case SDLK_v:
            min_distance += 0.001f;
        break;

        case SDLK_t:
            avoid_factor -= 0.01f;
        break;

        case SDLK_y:
            avoid_factor += 0.01f;
        break;   

        case SDLK_g:
            matching_factor -= 0.01f;
        break;

        case SDLK_h:
            matching_factor += 0.01f;
        break; 

        case SDLK_m:
            avoid_mouse = !avoid_mouse;
        break; 

        case SDLK_p:
            set_default();
        break; 

        default:
            return false;      
    }
    return true;
}