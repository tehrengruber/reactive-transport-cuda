#include <boost/fiber/all.hpp>

#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

void fn() {
    asm volatile("" : : : "memory");
}

void fn2() {
    PROFILER_START("fiber_wait");

    PROFILER_STOP("fiber_wait");
}

int main() {
    // create fiber
    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("fiber_creation");
        boost::fibers::fiber f1( fn );
        f1.join();
        PROFILER_STOP("fiber_creation");
    }

    // create fiber2
    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("fiber_creation2");
        boost::fibers::fiber f1( fn );
        f1.join();
        PROFILER_STOP("fiber_creation2");
    }

    // thread
    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("thread_creation");
        std::thread t1( fn );
        t1.join();
        PROFILER_STOP("thread_creation");
    }

    // fiber future
    for (size_t i=0; i<10000; ++i) {
        PROFILER_START("fiber_future_create");
        boost::fibers::promise<int> prom;
        boost::fibers::future<int> fut = prom.get_future();
        PROFILER_STOP("fiber_future_create");
        boost::fibers::fiber f([&fut]{
            PROFILER_START("fiber_future");
            fut.get();
            PROFILER_STOP("fiber_future");
        });
        PROFILER_START("fiber_set_value");
        prom.set_value(1);
        PROFILER_STOP("fiber_set_value");
        f.join();
    }

    // condition variable
    for (size_t i=0; i<10000; ++i) {
        std::mutex m;
        std::condition_variable cond_var;
        bool done = false;

        std::thread f([&]{
            PROFILER_START("std_condition_var");
            std::unique_lock<std::mutex> lock(m);
            while (!done) {
                cond_var.wait(lock);
            }
            PROFILER_STOP("std_condition_var");
        });

        {
            std::unique_lock<std::mutex> lock(m);
            done = true;
            cond_var.notify_one();
        }

        f.join();
    }

    //
    for (size_t i=0; i<10000; ++i) {
        std::mutex m;
        std::condition_variable cond_var;
        bool done = false;

        boost::fibers::fiber f([&]{
            PROFILER_START("fiber_condition_var");
            std::unique_lock<std::mutex> lock(m);
            cond_var.wait(lock);
            PROFILER_STOP("fiber_condition_var");
        });

        cond_var.notify_all();

        f.join();
    }

    Profiler::IO::print_timings();
}