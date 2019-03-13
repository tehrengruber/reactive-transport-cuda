#ifndef REACTIVETRANSPORTGPU_GAUSS_PARTIAL_PIV_H
#define REACTIVETRANSPORTGPU_GAUSS_PARTIAL_PIV_H

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

#include <Eigen/Dense>

#include <cfenv>

#include "common.h"

#define IDX(i, j) i+j*n

template<typename T>
DEVICE_DECL_SPEC
void custom_swap(T& t1, T& t2) {
    T temp = std::move(t1); // or T temp(std::move(t1));
    t1 = std::move(t2);
    t2 = std::move(temp);
}

template <typename T, size_t n>
DEVICE_DECL_SPEC
void gauss(T* lu, T* b, T* x) {
    size_t indx[n];
    indx[n-1]=n-1;

    for(int k = 0; k < n-1; ++k) {
        int row_of_biggest_in_col=k;
        double biggest_in_corner = std::abs(lu[IDX(k, k)]);
        for (size_t i=k+1; i<n; ++i) {
            double tmp=std::abs(lu[IDX(i, k)]);
            if (tmp > biggest_in_corner) {
                row_of_biggest_in_col=i;
                biggest_in_corner = tmp;
            }
        }
        indx[k] = row_of_biggest_in_col;

        if(biggest_in_corner != 0) {
            if (k != row_of_biggest_in_col) {
                for (size_t j = 0; j < n; ++j) {
                    custom_swap(lu[IDX(k, j)], lu[IDX(row_of_biggest_in_col, j)]);
                }
            }

            for (int i = k + 1; i < n; ++i) {
                lu[IDX(i, k)] /= lu[IDX(k, k)];
            }
        } else {
            assert(false);
        }
        if (k<n-1) {
            for (int i=k+1; i<n; ++i) {
                for (int j=k+1; j<n; ++j) {
                    lu[IDX(i, j)] -= lu[IDX(i, k)]*lu[IDX(k, j)];
                }
            }
        }
    }

    /*
     * Solve LSE
     */
    for (int i=0;i<n;i++)
        x[i] = b[i];
    for (int i=0;i<n;i++) {
        custom_swap(x[indx[i]], x[i]);
    }
    for (int i=0;i<n;i++) {
        for (int k=i+1; k<n; ++k) {
            x[k] -= x[i] * lu[IDX(k, i)];
        }
    }
    for (int i=n-1;i>=0;i--) {
        x[i] /= lu[IDX(i, i)];
        for (int k=0; k<i; ++k) {
            x[k] -= x[i] * lu[IDX(k, i)];
        }
    }
}

template <typename T>
DEVICE_DECL_SPEC
void gauss_ds(size_t n, T* lu, T* b, T* x) {
    size_t indx[n];
    indx[n-1]=n-1;

    for(int k = 0; k < n-1; ++k) {
        int row_of_biggest_in_col=k;
        double biggest_in_corner = std::abs(lu[IDX(k, k)]);
        for (size_t i=k+1; i<n; ++i) {
            double tmp=std::abs(lu[IDX(i, k)]);
            if (tmp > biggest_in_corner) {
                row_of_biggest_in_col=i;
                biggest_in_corner = tmp;
            }
        }
        indx[k] = row_of_biggest_in_col;

        if(biggest_in_corner != 0) {
            if (k != row_of_biggest_in_col) {
                for (size_t j = 0; j < n; ++j) {
                    custom_swap(lu[IDX(k, j)], lu[IDX(row_of_biggest_in_col, j)]);
                }
            }

            for (int i = k + 1; i < n; ++i) {
                lu[IDX(i, k)] /= lu[IDX(k, k)];
            }
        } else {
            assert(false);
        }
        if (k<n-1) {
            for (int i=k+1; i<n; ++i) {
                for (int j=k+1; j<n; ++j) {
                    lu[IDX(i, j)] -= lu[IDX(i, k)]*lu[IDX(k, j)];
                }
            }
        }
    }

    /*
     * Solve LSE
     */
    for (int i=0;i<n;i++)
        x[i] = b[i];
    for (int i=0;i<n;i++) {
        custom_swap(x[indx[i]], x[i]);
    }
    for (int i=0;i<n;i++) {
        for (int k=i+1; k<n; ++k) {
            x[k] -= x[i] * lu[IDX(k, i)];
        }
    }
    for (int i=n-1;i>=0;i--) {
        x[i] /= lu[IDX(i, i)];
        for (int k=0; k<i; ++k) {
            x[k] -= x[i] * lu[IDX(k, i)];
        }
    }
}

#endif