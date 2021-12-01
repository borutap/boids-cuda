#include "logger.h"
#include <iomanip>
#include <ctime>
#include <sstream>
#include <chrono>
#include <iostream>

Logger::Logger()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << "debug_" << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << ".csv";
    auto title = oss.str();
    file_handle.open(title);
    file_handle << "Measurement,Time (s)\n";
}

void Logger::start_timed_measurement(std::string title)
{
    file_handle << title << ",";
    start = std::chrono::high_resolution_clock::now();
}

void Logger::end_timed_measurement()
{
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    // std::string s_start = format_time(start);
    // std::string s_end = format_time(end);
    // auto delta = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::string s_delta = format_time(delta);
    file_handle << elapsed_seconds.count() << "\n";//," << s_delta << "\n";
}

void Logger::close_file()
{
    file_handle.close();
}

std::string Logger::format_time(std::chrono::_V2::system_clock::time_point &t)
{
    auto in_time_t = std::chrono::system_clock::to_time_t(t);

    return std::ctime(&in_time_t);
}
