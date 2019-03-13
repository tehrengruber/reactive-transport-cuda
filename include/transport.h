#ifndef REACTIVETRANSPORTGPU_TRANSPORT_H
#define REACTIVETRANSPORTGPU_TRANSPORT_H

#include "common.h"

// Perform a transport step
void transport(Vector<numeric_t, Eigen::Dynamic>& u, numeric_t dt, numeric_t dx, numeric_t v, numeric_t D, numeric_t ul);

#endif //REACTIVETRANSPORTGPU_TRANSPORT_H
