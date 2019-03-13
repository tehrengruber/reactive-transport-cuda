#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "cublas_v2.h"
#include <Eigen/Dense>
#include "magma_v2.h"

#include "common_cuda.h"
#include "../src/kernel/gauss_partial_pivoting_kernel.cu"

#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

void measure_gpu_memory_bandwidth() {
    size_t num_bytes = size_t(4)*1024*1024*128;

    std::linear_congruential_engine<std::uint_fast32_t, 16807, 0, 2147483647> eng;

    // allocate and set host memory
    uint8_t* buffer = new uint8_t[num_bytes];
    for (size_t i=0; i<num_bytes; ++i) {
        buffer[i] = (uint8_t) eng();
    }

    // allocate device memory
    void* dev_ptr;
    if (cudaMalloc(&dev_ptr, num_bytes*sizeof(uint8_t)) != cudaSuccess) {
        printf ("device memory allocation failed");
        return;
    }

    // measure host to device transfer speed
    double host_to_device_speed;
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (cudaMemcpy(dev_ptr, buffer, num_bytes*sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            printf ("device memory transfer failed");
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_seconds = end-start;
        host_to_device_speed = num_bytes/elapsed_seconds.count();
    }

    double device_to_host_speed;
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (cudaMemcpy(buffer, dev_ptr, num_bytes*sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf ("device memory transfer failed");
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_seconds = end-start;
        device_to_host_speed = num_bytes/elapsed_seconds.count();
    }

    if (cudaFreeHost(dev_ptr) != cudaSuccess) {
        printf ("device memory deallocation failed");
        return;
    }

    std::cout << "  Host to Device Transfer Rate: " << std::round(100*(host_to_device_speed)/1e9)/100 << "GB/s" << std::endl;
    std::cout << "  Device to Host Transfer Rate: " << std::round(100*(device_to_host_speed)/1e9)/100 << "GB/s" << std::endl;
}

void print_cuda_devices() {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Cuda Devices" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        // General properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

        // Memory size
        size_t free;
        size_t total;
        cudaMemGetInfo(&free, &total);
        printf("  Total memory (GB): %f\n",
               double(total)/(1024*1024*1024));

        // Measured memory bandwidth
        cudaSetDevice(i);
        //measure_gpu_memory_bandwidth();
    }
    std::cout << "------------------------------------------------" << std::endl;

    // make sure that the default device is selected again
    cudaSetDevice(0);
}

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

// todo: cudedevicereset on error

struct CuBLASLUSolver {
    size_t n;
    size_t batch_size;

    numeric_t* As;
    numeric_t* bs;

    // pointer to device memory containing lhs matrices
    numeric_t* device_ptr_As;
    // pointer to device memory containing rhs vectors
    numeric_t* device_ptr_bs;
    // pointer to device memory containing pivoting sequence of each factorization
    int* device_ptr_pivot_arr;
    // pointer to device memory containing information about the success of the decompositions
    int* device_ptr_info_arr;

    numeric_t** device_ptrs_As;
    numeric_t** device_ptrs_bs;

    cublasHandle_t handle;

    float runtime_getrs = 0;
    float runtime_getrf = 0;

    CuBLASLUSolver(const size_t n_, const size_t batch_size_) : n(n_), batch_size(batch_size_) {
        // allocate host memory
        As = new numeric_t[batch_size*n*n];
        bs = new numeric_t[batch_size*n];

        // allocate device memory
        if (cudaMalloc((void**) &device_ptr_As, batch_size*n*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_bs, batch_size*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_pivot_arr, batch_size*n*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_As, batch_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_bs, batch_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_info_arr, batch_size*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        // create an array containing the (device) addresses to As and bs
        numeric_t** ptrs_As = new numeric_t*[batch_size];
        numeric_t** ptrs_bs = new numeric_t*[batch_size];
        for (size_t i=0; i<batch_size; ++i) {
            ptrs_As[i] = device_ptr_As + i*n*n;
            ptrs_bs[i] = device_ptr_bs + i*n;
        }

        if (cudaMemcpy(device_ptrs_As, ptrs_As, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptrs_bs, ptrs_bs, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");

        // init cuBLAS
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }
    }

    ~CuBLASLUSolver() {
        if (cudaFree((void*) device_ptr_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_pivot_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_info_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
    }

    void solve() {
        /*
         * transfer data to the GPU
         */
        if (cudaMemcpy(device_ptr_As, As, batch_size*n*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptr_bs, bs, batch_size*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        //if (cudaMemset(device_ptr_pivot_arr, 0, batch_size*n*sizeof(int)) != cudaSuccess)
        //    throw std::runtime_error("Device memory initialization failed");

        cudaEvent_t start_getrf;
        cudaEvent_t stop_getrf;
        cudaEventCreate(&start_getrf);
        cudaEventCreate(&stop_getrf);
        cudaEvent_t start_getrs;
        cudaEvent_t stop_getrs;
        cudaEventCreate(&start_getrs);
        cudaEventCreate(&stop_getrs);

        /*
         * perform LU factorization
         */
        cudaEventRecord(start_getrf);
        int fac_res = cublasDgetrfBatched(handle, n, device_ptrs_As, n, device_ptr_pivot_arr, device_ptr_info_arr, batch_size);
        cudaEventRecord(stop_getrf);
        if (fac_res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("LU factorization failed");

        /*
         * solve LSE's
         */
        int info;
        cudaEventRecord(start_getrs);
        int sol_res = cublasDgetrsBatched(handle, CUBLAS_OP_N, n, 1, const_cast<const numeric_t **>(device_ptrs_As), n, device_ptr_pivot_arr, device_ptrs_bs, n, &info, batch_size);
        cudaEventRecord(stop_getrs);
        if (sol_res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("LU solve failed");

        cudaDeviceSynchronize();

        cudaEventSynchronize(stop_getrf);
        cudaEventSynchronize(stop_getrs);
        cudaEventElapsedTime(&runtime_getrf, start_getrf, stop_getrf);
        runtime_getrf/=1000;
        cudaEventElapsedTime(&runtime_getrs, start_getrs, stop_getrs);
        runtime_getrs/=1000;

        /*
         * transfer data back to host
         */
        if (cudaMemcpy(bs, device_ptr_bs, batch_size*n*sizeof(numeric_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Host memory transfer failed");
    }
};

struct MagmaLUSolver {
    size_t n;
    size_t batch_size;

    numeric_t* As;
    numeric_t* bs;

    // pointer to device memory containing lhs matrices
    numeric_t* device_ptr_As;
    // pointer to device memory containing rhs vectors
    numeric_t* device_ptr_bs;
    // pointer to device memory containing pivoting sequence of each factorization
    int* device_ptr_pivot_arr;
    // pointer to device memory containing information about the success of the decompositions
    int* device_ptr_info_arr;

    numeric_t** device_ptrs_As;
    numeric_t** device_ptrs_bs;

    cublasHandle_t handle;

    double total_runtime = 0;

    magma_queue_t magma_queue;

    magma_int_t* magma_pivot_arr;
    magma_int_t** magma_pivot_arr_ptrs;

    MagmaLUSolver(const size_t n_, const size_t batch_size_) : n(n_), batch_size(batch_size_) {
        // allocate host memory
        As = new numeric_t[batch_size*n*n];
        bs = new numeric_t[batch_size*n];

        // allocate device memory
        if (cudaMalloc((void**) &device_ptr_As, batch_size*n*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_bs, batch_size*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_pivot_arr, batch_size*n*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_As, batch_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_bs, batch_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_info_arr, batch_size*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        // create an array containing the (device) addresses to As and bs
        numeric_t** ptrs_As = new numeric_t*[batch_size];
        numeric_t** ptrs_bs = new numeric_t*[batch_size];
        for (size_t i=0; i<batch_size; ++i) {
            ptrs_As[i] = device_ptr_As + i*n*n;
            ptrs_bs[i] = device_ptr_bs + i*n;
        }

        if (cudaMemcpy(device_ptrs_As, ptrs_As, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptrs_bs, ptrs_bs, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");

        // init magma
        magma_init();
        int device;
        cudaGetDevice(&device);
        magma_queue_create(device, &magma_queue);

        // allocate pivot arrays
        if (cudaMalloc((void**) &magma_pivot_arr, batch_size*n*sizeof(magma_int_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        if (cudaMalloc((void**) &magma_pivot_arr_ptrs, batch_size*sizeof(magma_int_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        magma_int_t** pivot_arr_ptrs = new magma_int_t*[batch_size];
        for (size_t i=0; i<batch_size; ++i) {
            pivot_arr_ptrs[i] = magma_pivot_arr+n*i;
        }

        if (cudaMemcpy((void*) magma_pivot_arr_ptrs, (void*) pivot_arr_ptrs, batch_size*sizeof(magma_int_t*), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer failed");
        }
        delete[] pivot_arr_ptrs;
    }

    ~MagmaLUSolver() {
        if (cudaFree((void*) device_ptr_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_pivot_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_info_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;

        cudaFree(magma_pivot_arr);
        cudaFree(magma_pivot_arr_ptrs);
        magma_queue_destroy(magma_queue);
        magma_finalize();
    }

    void solve() {
        /*
         * transfer data to the GPU
         */
        if (cudaMemcpy(device_ptr_As, As, batch_size*n*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptr_bs, bs, batch_size*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        //if (cudaMemset(device_ptr_pivot_arr, 0, batch_size*n*sizeof(int)) != cudaSuccess)
        //    throw std::runtime_error("Device memory initialization failed");

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        /*
         * LU solve
         */
        cudaEventRecord(start, magma_queue_get_cuda_stream(magma_queue));

        magma_dgesv_batched(n, 1, device_ptrs_As, n, magma_pivot_arr_ptrs, device_ptrs_bs, n, device_ptr_info_arr, batch_size, magma_queue);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float runtime = 0;
        cudaEventElapsedTime(&runtime, start, stop);
        total_runtime = runtime/1000;

        cudaDeviceSynchronize();

        /*
         * transfer data back to host
         */
        if (cudaMemcpy(bs, device_ptr_bs, batch_size*n*sizeof(numeric_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Host memory transfer failed");
    }
};

struct OverlappingBatchLUSolver {
    size_t n;
    size_t batch_size;
    size_t queue_size;

    numeric_t* As;
    numeric_t* bs;

    // pointer to device memory containing lhs matrices
    numeric_t* device_ptr_As;
    // pointer to device memory containing rhs vectors
    numeric_t* device_ptr_bs;
    // pointer to device memory containing pivoting sequence of each factorization
    int* device_ptr_pivot_arr;
    // pointer to device memory containing information about the success of the decompositions
    int* device_ptr_info_arr;

    numeric_t** device_ptrs_As;
    numeric_t** device_ptrs_bs;

    cublasHandle_t handle;

    cudaStream_t* streams;

    OverlappingBatchLUSolver(const size_t n_, const size_t queue_size_, const size_t batch_size_) : n(n_), batch_size(batch_size_), queue_size(queue_size_) {
        assert(queue_size_%batch_size==0);

        streams = new cudaStream_t[queue_size/batch_size];
        for (size_t i=0; i<queue_size/batch_size; ++i) {
            cudaStreamCreate(&streams[i]);
        }

        // allocate host memory
        if (cudaMallocHost((void**) &As, queue_size*n*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Host memory allocation failed");
        if (cudaMallocHost((void**) &bs, queue_size*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Host memory allocation failed");

        // allocate device memory
        if (cudaMalloc((void**) &device_ptr_As, queue_size*n*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_bs, queue_size*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_pivot_arr, queue_size*n*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_As, queue_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptrs_bs, queue_size*sizeof(numeric_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_ptr_info_arr, queue_size*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        // create an array containing the (device) addresses to As and bs
        numeric_t** ptrs_As = new numeric_t*[queue_size];
        numeric_t** ptrs_bs = new numeric_t*[queue_size];
        for (size_t i=0; i<queue_size; ++i) {
            ptrs_As[i] = device_ptr_As + i*n*n;
            ptrs_bs[i] = device_ptr_bs + i*n;
        }

        if (cudaMemcpy(device_ptrs_As, ptrs_As, queue_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptrs_bs, ptrs_bs, queue_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");

        // init cuBLAS
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }
    }

    ~OverlappingBatchLUSolver() {
        if (cudaFree((void*) device_ptr_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_pivot_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptrs_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_ptr_info_arr) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;

        if (cudaFreeHost((void*) As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFreeHost((void*) bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
    }

    void solve() {
        for (size_t i=0; i<queue_size/batch_size; ++i) {
            numeric_t* device_ptr_As_prime = device_ptr_As+i*batch_size*n*n;
            numeric_t* device_ptr_bs_prime = device_ptr_bs+i*batch_size*n;
            numeric_t** device_ptrs_As_prime = device_ptrs_As+i*batch_size;
            numeric_t** device_ptrs_bs_prime = device_ptrs_bs+i*batch_size;
            int* device_ptr_pivot_arr_prime = device_ptr_pivot_arr+i*batch_size*n;
            int* device_ptr_info_arr_prime = device_ptr_info_arr+i*batch_size;

            // set cuBLAS stream
            cublasSetStream(handle, streams[i]);

            /*
             * transfer data to the GPU
             */
            if (cudaMemcpyAsync(device_ptr_As_prime, As+i*batch_size*n*n, batch_size*n*n*sizeof(numeric_t), cudaMemcpyHostToDevice, streams[i]) != cudaSuccess)
                throw std::runtime_error("Device memory transfer failed");
            if (cudaMemcpyAsync(device_ptr_bs_prime, bs+i*batch_size*n, batch_size*n*sizeof(numeric_t), cudaMemcpyHostToDevice, streams[i]) != cudaSuccess)
                throw std::runtime_error("Device memory transfer failed");

            /*
             * perform LU factorization
             */
            int fac_res = cublasDgetrfBatched(handle, n, device_ptrs_As_prime, n, device_ptr_pivot_arr_prime, device_ptr_info_arr_prime, batch_size);
            if (fac_res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("LU factorization failed");

            /*
             * solve LSE's
             */
            int info;
            int sol_res = cublasDgetrsBatched(handle, CUBLAS_OP_N, n, 1, const_cast<const numeric_t **>(device_ptrs_As_prime), n, device_ptr_pivot_arr_prime, device_ptrs_bs_prime, n, &info, batch_size);

            if (sol_res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("LU solve failed");

            /*
             * transfer data back to host
             */
            if (cudaMemcpyAsync(bs+i*batch_size*n, device_ptr_bs_prime, batch_size*n*sizeof(numeric_t), cudaMemcpyDeviceToHost, streams[i]) != cudaSuccess)
                throw std::runtime_error("Host memory transfer failed");
        }

        for (size_t i=0; i<queue_size/batch_size; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    }
};

struct OwnLUSolver {
    size_t n;
    size_t number_of_lses;

    double total_runtime = 0;

    numeric_t* As;
    numeric_t* bs;

    numeric_t* device_As;
    numeric_t* device_bs;
    numeric_t* device_xs;

    OwnLUSolver(size_t n_, const size_t number_of_lses_) : n(n_), number_of_lses(number_of_lses_) {
        As = new numeric_t[n*n*number_of_lses];
        bs = new numeric_t[n*number_of_lses];
        if (cudaMalloc((void**) &device_As, number_of_lses*n*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_bs, number_of_lses*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_xs, number_of_lses*n*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
    }

    void solve() {
        if (cudaMemcpy(device_As, As, number_of_lses*n*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_bs, bs, number_of_lses*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");

        CudaExecutionConfiguration conf_gpv("gauss_partial_pivoting_kernel", (void*) gauss_partial_pivoting_kernel, number_of_lses);

        SimpleTimer timer;
        timer.tic();
        gauss_partial_pivoting_kernel<<<conf_gpv.grid_size, conf_gpv.block_size>>>(number_of_lses, device_As, device_bs, device_xs);
        cudaDeviceSynchronize();
        timer.toc();
        total_runtime = timer.duration();
    }

    ~OwnLUSolver() {
        if (cudaFree((void*) device_As) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_bs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
        if (cudaFree((void*) device_xs) != cudaSuccess)
            std::cerr << "Device memory deallocation failed" << std::endl;
    }
};

template <size_t n>
void initialize_solver(size_t num_lse, Eigen::Matrix<numeric_t, n, n> A, Eigen::Matrix<numeric_t, n, 1> b, numeric_t* As, numeric_t* bs) {
    for (size_t i=0; i<num_lse; ++i) {
        numeric_t* A_ = As+n*n*i;
        numeric_t* b_ = bs+n*i;
        for (size_t j=0; j<n*n; ++j) {
            A_[j] = *(A.data()+j);
        }
        for (size_t j=0; j<n; ++j) {
            b_[j] = *(b.data()+j);
        }
    }
}

double num_flops(size_t n) {
    return 2./3*std::pow(n, 3);
}

int main () {
    // Print device information
    print_cuda_devices();

    size_t number_of_lses = 10000;

    // Parameters
    constexpr size_t n = 100;

    Eigen::Matrix<numeric_t, n, n> A;
    Eigen::Matrix<numeric_t, n, 1> b;
    Eigen::Matrix<numeric_t, n, 1> x;

    /*A <<    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -2,    -1,    -0,    -0,    -0,    -0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -1,    -0,    -0,    -0,    -0,    -0,     0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -1,    -1,    -0,    -0,    -0,    -0,     0,     0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -1,    -3,    -1,    -0,    -0,    -0,     0,     0,     0,    -1,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -3,    -1,    -0,    -0,    -0,     0,     0,     0,     0,    -1,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -2,    -1,    -0,    -0,    -0,     0,     0,     0,     0,     0,    -1,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -0,    -0,    -1,    -0,    -0,     0,     0,     0,     0,     0,     0,    -1,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -0,    -0,    -0,    -1,    -0,     0,     0,     0,     0,     0,     0,     0,    -1,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -0,    -0,    -0,    -0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,    -1,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -2,    -1,    -0,    -0,    -0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -1,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -0,    -3,    -1,    -0,    -0,    -1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    -1,
            2,     1,     1,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            1,     0,     1,     3,     3,     2,     0,     0,     0,     2,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     1,     1,     1,     0,     0,     0,     1,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 1e-14;*/

    A.setIdentity();

    b << 87.5046,
            1,
            57.5729,
            214.027,
            190.798,
            141.883,
            96.3161,
            48.9623,
            199.894,
            146.101,
            409.757,
            111.016,
            55.608,
            0.05,
            9.999e-11,
            9.999e-11,
            9.998e-11,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0,
            -0;

    {
        CuBLASLUSolver solver(n, number_of_lses);
        initialize_solver<n>(number_of_lses, A, b, solver.As, solver.bs);
        solver.solve();

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "cuBLAS" << std::endl;
        std::cout << " time dgetrf: " << solver.runtime_getrf << std::endl;
        std::cout << " time dgetrs: " << solver.runtime_getrs << std::endl;
        std::cout << " time total: "  << solver.runtime_getrf+solver.runtime_getrs << std::endl;
        std::cout << " performance: " << 1e-9*number_of_lses*num_flops(n)/(solver.runtime_getrf+solver.runtime_getrs) << std::endl;
    }
    {
        MagmaLUSolver solver(n, number_of_lses);
        initialize_solver<n>(number_of_lses, A, b, solver.As, solver.bs);
        solver.solve();

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "magma" << std::endl;
        std::cout << " time total: "  << solver.total_runtime << std::endl;
        std::cout << " performance: " << 1e-9*number_of_lses*num_flops(n)/(solver.total_runtime) << std::endl;
    }
    {
        OwnLUSolver solver(n, number_of_lses);
        initialize_solver<n>(number_of_lses, A, b, solver.As, solver.bs);
        solver.solve();

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "own" << std::endl;
        std::cout << " time total: "  << solver.total_runtime << std::endl;
        std::cout << " performance: " << 1e-9*number_of_lses*num_flops(n)/(solver.total_runtime) << std::endl;
    }
    {
        SimpleTimer timer;
        timer.tic();
        for (size_t i=0; i<1; ++i) {
            for (size_t i=0; i<number_of_lses; ++i) {
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
            }
        }
        timer.toc();

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "eigen" << std::endl;
        std::cout << " time total: "  << timer.duration() << std::endl;
        std::cout << " performance: " << 1e-9*number_of_lses*num_flops(n)/(timer.duration()) << std::endl;
    }
}