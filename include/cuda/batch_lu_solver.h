#ifndef REACTIVETRANSPORTGPU_BATCH_LU_SOLVER_H
#define REACTIVETRANSPORTGPU_BATCH_LU_SOLVER_H
#include "common.h"
#include "cublas_v2.h"

struct BatchLUSolver {
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

    numeric_t** device_ptrs_As;
    numeric_t** device_ptrs_bs;

    double runtime;

    BatchLUSolver(const size_t n_, const size_t batch_size_) : n(n_), batch_size(batch_size_) {
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

        // create an array containing the (device) addresses to As and bs
        numeric_t** ptrs_As = new numeric_t*[batch_size];
        numeric_t** ptrs_bs = new numeric_t*[batch_size];
        for (size_t i=0; i<batch_size; ++i) {
            ptrs_As[i] = device_ptr_As + i*n*n;
            ptrs_bs[i] = device_ptr_bs + i*n;
        }

        if (cudaMemcpy(device_ptrs_As, ptrs_As, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transer failed");
        if (cudaMemcpy(device_ptrs_bs, ptrs_bs, batch_size*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transer failed");
    }

    ~BatchLUSolver() {
        cudaFree(device_ptr_As);
        cudaFree(device_ptr_bs);
        cudaFree(device_ptr_pivot_arr);
        cudaFree(device_ptrs_As);
        cudaFree(device_ptrs_bs);
        delete[] As;
        delete[] bs;
    }

    void solve() {
        cublasHandle_t handle;
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        cudaEventRecord(start_event);
        cudaEventRecord(stop_event);

        /*
         * initialize cuBLAS
         */
        {
            cublasStatus_t stat;
            stat = cublasCreate(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS initialization failed");
            }
        }

        /*
         * transfer data to the GPU
         */
        if (cudaMemcpy(device_ptr_As, As, batch_size*n*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        if (cudaMemcpy(device_ptr_bs, bs, batch_size*n*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Device memory transfer failed");
        //if (cudaMemset(device_ptr_pivot_arr, 0, batch_size*n*sizeof(int)) != cudaSuccess)
        //    throw std::runtime_error("Device memory initialization failed");



        /*
         * perform LU factorization
         */
        int* device_ptr_info_arr;

        if (cudaMalloc((void**) &device_ptr_info_arr, batch_size*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        //if (cudaMemset(device_ptr_info_arr, 0, batch_size*sizeof(int)) != cudaSuccess)
        //    throw std::runtime_error("Device memory initialization failed");

        cudaEventRecord(start_event);
        int fac_res = cublasDgetrfBatched(handle, n, device_ptrs_As, n, device_ptr_pivot_arr, device_ptr_info_arr, batch_size);
        if (fac_res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("LU factorization failed");

        if (cudaMemcpy(As, device_ptr_As, n*n*sizeof(numeric_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Host memory transfer failed");

        cudaFree(device_ptr_info_arr);

        /*
         * solve LSE's
         */
        int info;
        int sol_res = cublasDgetrsBatched(handle, CUBLAS_OP_N, n, 1, const_cast<const numeric_t **>(device_ptrs_As), n, device_ptr_pivot_arr, device_ptrs_bs, n, &info, batch_size);
        if (sol_res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("LU solve failed");
        cudaEventRecord(stop_event);

        /*
         * transfer data back to host
         */
        if (cudaMemcpy(bs, device_ptr_bs, batch_size*n*sizeof(numeric_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Host memory transfer failed");

        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        runtime = milliseconds/1000;
    }
};

#endif //REACTIVETRANSPORTGPU_BATCH_LU_SOLVER_H
