//
// Created by Khanh Nguyen Ngoc on 13/4/21.
//

#ifndef HIGH_PRECISION_TIMER_HIGH_PRECISION_TIMER_H
#define HIGH_PRECISION_TIMER_HIGH_PRECISION_TIMER_H
#include<cstdint>
#include<ctime>
namespace timer {
    int64_t now();

    #ifdef _WIN32
        #include<Windows.h>
        int64_t freq = 0;
        int64_t now() {
            if (0 == freq) {
                    QueryPerformanceFrequency((LARGE_INTEGER*)&m_freq);
            }
            int64_t time_point;
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            time_point = 1000000000*t.QuadPart/freq;
            return time_point;
        }
    #else
        #include <ctime>
        int64_t now() {
            timespec tp{};
            clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
            int64_t time_point = static_cast<int64_t>(1000000000 * tp.tv_sec) + static_cast<int64_t>(tp.tv_nsec);
            return time_point;
        }
    #endif
}




#endif //HIGH_PRECISION_TIMER_HIGH_PRECISION_TIMER_H
