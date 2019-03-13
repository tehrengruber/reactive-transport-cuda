#ifndef REACTIVETRANSPORTGPU_SIMPLE_CUDA_PROFILER_H
#define REACTIVETRANSPORTGPU_SIMPLE_CUDA_PROFILER_H

#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

struct SimpleCudaProfiler {
    // maps name of each profiled sections to an index
    std::map<std::string, size_t> section_name_map;

    size_t next_section_id = 0;

    cudaEvent_t* start_events = nullptr;
    cudaEvent_t* stop_events = nullptr;

    std::vector<double> durations;
    std::vector<size_t> calls;
    std::vector<bool> valid;

    SimpleCudaProfiler() {}

    ~SimpleCudaProfiler() {
        for (size_t i=0; i<next_section_id; ++i) {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(stop_events[i]);
        }
        delete[] start_events;
        delete[] stop_events;
    }

    SimpleCudaProfiler(const SimpleCudaProfiler&) = delete;
    SimpleCudaProfiler& operator=(const SimpleCudaProfiler&) = delete;

    void add(std::string s) {
        size_t id = next_section_id++;
        section_name_map[s] = id;
    }

    void initialize() {
        if (start_events!=nullptr)
            delete[] start_events;
        if (stop_events!=nullptr)
            delete[] stop_events;
        start_events = new cudaEvent_t[next_section_id];
        stop_events = new cudaEvent_t[next_section_id];

        for (size_t i=0; i<next_section_id; ++i) {
            gpuErrchk(cudaEventCreate(&start_events[i]));
            gpuErrchk(cudaEventCreate(&stop_events[i]));
            durations.push_back(0);
            calls.push_back(0);
            valid.push_back(false);
        }
    }

    void start(std::string section_name) {
        size_t i = section_name_map[section_name];
        gpuErrchk(cudaEventRecord(start_events[i]));
    }

    void stop(std::string section_name) {
        size_t i = section_name_map[section_name];
        gpuErrchk(cudaEventRecord(stop_events[i]));
        valid[i] = true;
    }

    void update() {
        for (size_t i=0; i<next_section_id; ++i) {
            if (valid[i]) {
                gpuErrchk(cudaEventSynchronize(stop_events[i]));
                float milliseconds = 0;
                gpuErrchk(cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]));

                durations[i] += milliseconds/1000;
                ++calls[i];
            }
            valid[i] = false;
        }
    }

    void print() {
        // compute total runtime
        double total_duration = 0;
        for (double duration : durations) {
            total_duration += duration;
        }

        // determine length of the name of the longest section
        size_t max_len = 0;
        for (auto& kv : section_name_map)
            max_len = std::max(kv.first.size(), max_len);

        // print
        std::cout << std::setw(max_len+1) << "section"
                  << std::setw(10) << "[s]"
                  << std::setw(10) << "[%]"
                  << std::setw(10) << "calls" << std::endl;
        for (auto& kv : section_name_map) {
            std::cout << std::setw(max_len+1) << kv.first
                      << std::setw(10) << std::round(100*durations[kv.second])/100
                      << std::setw(10) << std::round(10000*durations[kv.second]/total_duration)/100
                      << std::setw(10) << calls[kv.second] << std::endl;
        }
    }
};

#endif //REACTIVETRANSPORTGPU_SIMPLE_CUDA_PROFILER_H
