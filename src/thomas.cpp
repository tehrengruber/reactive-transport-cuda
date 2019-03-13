#include "common.h"
#include "thomas.h"

// Solve a tridiagonal matrix equation using Thomas algorithm.
// TODO: references?
Vector<numeric_t, Eigen::Dynamic> thomas(Vector<numeric_t, Eigen::Dynamic>& a, Vector<numeric_t, Eigen::Dynamic>& b,
                                         Vector<numeric_t, Eigen::Dynamic>& c, Vector<numeric_t, Eigen::Dynamic>& d) {
    size_t n = d.size();
    c[0] /= b[0];
    for (size_t i=1; i < n-1; ++i) {
        c[i] /= b[i] - a[i]*c[i - 1];
    }
    d[0] /= b[0];
    for (size_t i=1; i<n; ++i) {
        d[i] = (d[i] - a[i]*d[i - 1])/(b[i] - a[i]*c[i - 1]);
    }
    auto& x = d;
    for (int i=n-2; i>=0; --i) {
        x[i] -= c[i]*x[i + 1];
    }
    return x;
}