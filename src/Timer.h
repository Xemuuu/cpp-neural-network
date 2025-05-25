#pragma once
#include <iostream>
#include <chrono>

struct Timer
{
    const char* m_text;

    Timer (const char* text)
    {
        m_start = std::chrono::high_resolution_clock::now();
        m_text = text;
    }
 
    ~Timer ()
    {
        std::chrono::duration<float> seconds = std::chrono::high_resolution_clock::now() - m_start;
        printf("%s%0.2f sekund\n", m_text, seconds.count());
    }
 
    std::chrono::high_resolution_clock::time_point m_start;
};