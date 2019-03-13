#include "common.h"

#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

int main() {
    volatile double vol_val[5] = { 1.1, 1.1, 1.1, 1.1, 1.1 };
    double val[5] = { vol_val[0], vol_val[1], vol_val[2], vol_val[3], vol_val[4]};
    double result[5] = {0, 0, 0, 0, 0};
    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("logarithm");
        clobber();
        result[0] = std::log(result[0]);
        result[1] = std::log(result[1]);
        result[2] = std::log(result[2]);
        result[3] = std::log(result[3]);
        result[4] = std::log(result[4]);
        escape(result);
        clobber();
        PROFILER_STOP("logarithm");
        val[0] = vol_val[0];
        val[1] = vol_val[1];
        val[2] = vol_val[2];
        val[3] = vol_val[3];
        val[4] = vol_val[4];
    }

    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("div");
        clobber();
        result[0] = result[0]/val[0];
        result[1] = result[1]/val[1];
        result[2] = result[2]/val[2];
        result[3] = result[3]/val[3];
        result[4] = result[4]/val[4];
        escape(result[0]);
        escape(result[1]);
        escape(result[2]);
        escape(result[3]);
        escape(result[4]);
        clobber();
        PROFILER_STOP("div");
        val[0] = vol_val[0];
        val[1] = vol_val[1];
        val[2] = vol_val[2];
        val[3] = vol_val[3];
        val[4] = vol_val[4];
    }

    Profiler::IO::print_timings();
}