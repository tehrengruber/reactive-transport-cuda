#ifndef REACTIVETRANSPORTGPU_THOMAS_H
#define REACTIVETRANSPORTGPU_THOMAS_H

#include "common.h"

// Solve a tridiagonal matrix equation using Thomas algorithm.
// TODO: references?
Vector<numeric_t, Eigen::Dynamic> thomas(Vector<numeric_t, Eigen::Dynamic>& a, Vector<numeric_t, Eigen::Dynamic>& b,
                                         Vector<numeric_t, Eigen::Dynamic>& c, Vector<numeric_t, Eigen::Dynamic>& d);

#endif //REACTIVETRANSPORTGPU_THOMAS_H
