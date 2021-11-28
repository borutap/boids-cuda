#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <chrono>

class Logger
{
public:
    Logger();
    void start_timed_measurement(std::string title);
    void end_timed_measurement();
    void close_file();
private:
    std::ofstream file_handle;
    std::string curr_row_title;
    std::chrono::_V2::system_clock::time_point start;
    std::string format_time(
        std::chrono::_V2::system_clock::time_point &t);
};

#endif