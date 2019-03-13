#include <mkl_lapacke.h>
#undef I

#include "misc/simple_timer.h"

int main() {
    size_t batch_size = 10000;

    for (size_t n=0; n<128; n+=2) {
        int nrhs = 1;
        int lda = n;
        int ldb = n;
        int info;

        using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        using vector_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        using ivector_t = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

        matrix_t A_proto(n,n);
        A_proto.setRandom();
        matrix_t b_proto(n, n);
        b_proto.setRandom();

        std::vector<matrix_t> As(batch_size, A_proto);
        std::vector<vector_t> bs(batch_size, b_proto);
        std::vector<ivector_t> ipiv(batch_size);

        SimpleTimer t;
        t.tic();
        for (size_t i=0; i<=batch_size; ++i) {
            dgesv( &n, &nrhs, As[i].data(), &lda, ipiv[i].data(), bs[i].data(), &ldb, &info );
        }
        t.toc();

        std::cout << n << 2./3*std::pow(n, 3)/timer.duration() << std::endl;
    }
}