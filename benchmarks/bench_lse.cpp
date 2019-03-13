#include <iostream>
#include <chrono>
#include <fstream>

//#include <lapacke.h>
#include <mkl_lapacke.h>
#undef I
#include <Eigen/Dense>

#include "gauss_partial_pivoting.h"

#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

using numeric_t = double;

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

int main(int argc, char* argv[]) {
    {
        std::ofstream f;
        f.open ("bench_lse_double_warm_cache.txt");
        {
            constexpr size_t n = 2;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 6;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 10;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 14;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 18;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 22;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 26;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 30;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 34;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 38;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 42;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 46;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 50;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 54;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 58;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 62;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 66;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 70;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 74;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 78;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 82;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 86;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 90;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 94;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 98;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        for (size_t n=101; n<1000; n+=10) {
            Profiler::reset();

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A_proto(n, n);
            A_proto.setRandom();

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(n, n);
            Eigen::Matrix<double, Eigen::Dynamic, 1> b(n);
            b.setConstant(0);

            Eigen::Matrix<double, Eigen::Dynamic, 1> x(n);

            Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> lu;

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        f.close();
    }
    {
        std::ofstream f;
        f.open ("bench_lse_float_warm_cache.txt");
        {
            constexpr size_t n = 2;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 6;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 10;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 14;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 18;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 22;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 26;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 30;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 34;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 38;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 42;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 46;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 50;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 54;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 58;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 62;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 66;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 70;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 74;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 78;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 82;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 86;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 90;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 94;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 98;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        for (size_t n=101; n<1000; n+=10) {
            Profiler::reset();

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A_proto(n, n);
            A_proto.setRandom();

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(n, n);
            Eigen::Matrix<float, Eigen::Dynamic, 1> b(n);
            b.setConstant(0);

            Eigen::Matrix<float, Eigen::Dynamic, 1> x(n);

            Eigen::PartialPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> lu;

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;

                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        f.close();
    }
    {
        std::ofstream f;
        f.open ("bench_lse_double_cold_cache.txt");
        {
            constexpr size_t n = 2;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 6;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 10;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 14;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 18;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 22;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 26;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 30;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 34;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 38;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 42;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 46;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 50;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 54;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 58;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 62;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 66;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 70;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 74;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 78;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 82;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 86;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 90;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 94;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 98;
            Profiler::reset();

            Eigen::Matrix<double, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<double, n, n> A;
            Eigen::Matrix<double, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<double, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        for (size_t n=101; n<1000; n+=10) {
            Profiler::reset();

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A_proto(n, n);
            A_proto.setRandom();

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(n, n);
            Eigen::Matrix<double, Eigen::Dynamic, 1> b(n);
            b.setConstant(0);

            Eigen::Matrix<double, Eigen::Dynamic, 1> x(n);

            Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> lu;

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        f.close();
    }
    {
        std::ofstream f;
        f.open ("bench_lse_float_cold_cache.txt");
        {
            constexpr size_t n = 2;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 6;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 10;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 14;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 18;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 22;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 26;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 30;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 34;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 38;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 42;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 46;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 50;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 54;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 58;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 62;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 66;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 70;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 74;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 78;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 82;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 86;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 90;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 94;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        {
            constexpr size_t n = 98;
            Profiler::reset();

            Eigen::Matrix<float, n, n> A_proto;
            A_proto.setRandom();

            Eigen::Matrix<float, n, n> A;
            Eigen::Matrix<float, n, 1> b(n);
            b.setConstant(0);
            Eigen::Matrix<float, n, 1> x(n);

            Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, n, n>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        for (size_t n=101; n<1000; n+=10) {
            Profiler::reset();

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A_proto(n, n);
            A_proto.setRandom();

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(n, n);
            Eigen::Matrix<float, Eigen::Dynamic, 1> b(n);
            b.setConstant(0);

            Eigen::Matrix<float, Eigen::Dynamic, 1> x(n);

            Eigen::PartialPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> lu;

            size_t repetitions = 10000;
            if (n > 10)
                repetitions /= 10;
            if (n > 100)
                repetitions /= 100;
            if (n > 1000)
                repetitions /= 100;

            for (size_t i=0; i<repetitions; ++i) {
                A = A_proto;
                clear_cpu_cache();
                PROFILER_START("lu_solve");
                Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>> lu(A);
                x += lu.solve(b);
                escape(A);
                escape(b);
                escape(x);
                PROFILER_STOP("lu_solve");
                x.normalize();
            }

            f << n << " " << Profiler::get_last_timing()->mean() << "; " << std::endl;
            f.flush();
            std::cout << "n: " << n << " " << x.norm() << std::endl;
        }
        f.close();
    }
}