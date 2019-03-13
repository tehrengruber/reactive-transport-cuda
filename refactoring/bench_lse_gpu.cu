#include "batch_lu_solver.h"

int main() {
    for (size_t n=1; n<200; ++n) {
        size_t number_of_lses = 1e8/(n*n*8);
        size_t repititions = 10000;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(n, n);
        A.setRandom();
        Eigen::Matrix<double, Eigen::Dynamic, 1> b(n);
        b.setConstant(0);

        BatchLUSolver solver(n, number_of_lses);
        for (size_t i=0; i<number_of_lses; ++i) {
            numeric_t* A_ = solver.As+28*28*i;
            numeric_t* b_ = solver.bs+28*i;
            for (size_t j=0; j<28*28; ++j) {
                A_[j] = *(A.data()+j);
            }
            for (size_t j=0; j<28; ++j) {
                b_[j] = *(b.data()+j);
            }
        }

        double total_runtime = 0;
        for (size_t i=0; i<1; ++i) {
            solver.solve();
            total_runtime += solver.runtime;
        }
        total_runtime /= repititions;
        std::cout << n << " " << total_runtime << ";" << std::endl;
        //std::cout << "BatchLUSolver time: " << timer.duration() << std::endl;
        std::cout << "BatchLUSolver Gflop/s: " << number_of_lses*2./3*std::pow(28,3)/total_runtime*1e-9 << std::endl;
    }
}