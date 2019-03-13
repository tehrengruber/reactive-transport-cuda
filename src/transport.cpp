#include "thomas.h"
#include "transport.h"

// Perform a transport step
void transport(Vector<numeric_t, Eigen::Dynamic>& u, numeric_t dt, numeric_t dx, numeric_t v, numeric_t D, numeric_t ul) {
    size_t ncells = u.size();

    numeric_t alpha = D*dt/(dx*dx);
    numeric_t beta = v*dt/dx;
    Vector<numeric_t, Eigen::Dynamic> a(ncells);
    a.setConstant(-beta - alpha);
    Vector<numeric_t, Eigen::Dynamic> b(ncells);
    b.setConstant(1 + beta + 2*alpha);
    Vector<numeric_t, Eigen::Dynamic> c(ncells);
    c.setConstant(-alpha);
    // b[0] = 1.0
    // c[0] = 0.0
    // u[0] = ul
    u[0] += beta*ul;
    b[0] = 1 + beta + alpha;
    b[ncells-1] = 1 + beta + alpha;
    thomas(a, b, c, u);
}