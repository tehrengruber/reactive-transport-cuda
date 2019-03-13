#ifndef REACTIVETRANSPORTGPU_SIMPLE_TIMER_H
#define REACTIVETRANSPORTGPU_SIMPLE_TIMER_H

#include <chrono>

struct SimpleTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    std::chrono::time_point<std::chrono::high_resolution_clock> stop;

    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    void toc() {
        stop = std::chrono::high_resolution_clock::now();
    }

    double duration() {
        return std::chrono::duration<double>(stop-start).count();
    }
};

#endif //REACTIVETRANSPORTGPU_SIMPLE_TIMER_H
