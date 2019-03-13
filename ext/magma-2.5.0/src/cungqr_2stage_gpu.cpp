/*
    -- MAGMA (version 2.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2019

       @generated from src/zungqr_2stage_gpu.cpp, normal z -> c, Wed Jan  2 14:18:50 2019

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    CUNGQR generates an M-by-N COMPLEX matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by CGEQRF_GPU.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix Q. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    @param[in,out]
    dA      COMPLEX array A on the GPU device,
            dimension (LDDA,N). On entry, the i-th column must contain
            the vector which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by CGEQRF_GPU in the first k
            columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    ldda    INTEGER
            The first dimension of the array A. LDDA >= max(1,M).

    @param[in]
    tau     COMPLEX array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by CGEQRF_GPU.

    @param[in]
    dT      COMPLEX work space array on the GPU device,
            dimension (MIN(M, N) )*NB.
            This must be the 6th argument of magma_cgeqrf_gpu
            [ note that if N here is bigger than N in magma_cgeqrf_gpu,
              the workspace requirement DT in magma_cgeqrf_gpu must be
              as specified in this routine ].

    @param[in]
    nb      INTEGER
            This is the block size used in CGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in DT.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_ungqr
*******************************************************************************/
extern "C" magma_int_t
magma_cungqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info)
{
    #define dA(a_1,a_2) (dA + (a_2)*(ldda) + (a_1))
    #define dT(a_1)     (dT + (a_1)*nb)

    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    
    magma_int_t  i__1, i__2, i__3;
    //magma_int_t lwork;
    magma_int_t i, ib, ki, kk;  //, iinfo;
    //magma_int_t lddwork = min(m, n);
    //magmaFloatComplex *work, *panel;
    magmaFloatComplex_ptr dwork;
    magma_int_t ldt=nb; // need to be an input parameter
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    //magma_queue_create( cdev, &queues[1] );

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (ldda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (n <= 0)
        return *info;

    if (MAGMA_SUCCESS != magma_cmalloc( &dwork, n*nb )) {
        printf ("!!!! cungqr_2stage magma_alloc failed for: dwork\n" );
        return MAGMA_ERR_ALLOCATION;
    }

    if ( (nb > 1) && (nb < k) ) {
        /*  Use blocked code after the last block.
            The first kk columns are handled by the block method.
            ki is start of 2nd-to-last block. */
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);

        /* Set A(1:kk,kk+1:n) to zero. */
        /* and A(kk+1:m, kk+1:n) = I */
        magmablas_claset( MagmaFull, kk,   n-kk, c_zero, c_zero, dA(0, kk), ldda, queues[0] );
        magmablas_claset( MagmaFull, m-kk, n-kk, c_zero, c_one,  dA(kk,kk), ldda, queues[0] );
    }
    else {
        ki = 0;
        kk = 0;
    }
    
    /* Allocate work space on CPU in pinned memory */
    //lwork = (n+m) * nb;
    //if (kk < n)
    //  lwork = max(lwork, n * nb + (m-kk)*(n-kk));

    //if (MAGMA_SUCCESS != magma_cmalloc_pinned( &work, (lwork) )) {
    //    *info = MAGMA_ERR_HOST_ALLOC;
    //    return *info;
    //}
    //panel = work + n * nb;

    /* Use unblocked code for the last or only block. */
    if (kk < n) {
        i__1 = m - kk;
        i__2 = n - kk;
        i__3 = k - kk;
        //magma_cgetmatrix( i__1, i__2, dA(kk, kk), ldda, panel, i__1, queues[0] );
        //lapackf77_cungqr(&i__1, &i__2, &i__3, panel, &i__1, &tau[kk],
        //                 work, &lwork, &iinfo);
        //
        //magma_csetmatrix( i__1, i__2, panel, i__1, dA(kk, kk), ldda, queues[0] );
        
        magma_clarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                          i__1, i__2, i__3,
                          dA(kk, kk-nb), ldda, dT(kk-nb), ldt,
                          dA(kk, kk), ldda, dwork, i__2, queues[0] );
        
        //magmablas_claset( MagmaFull, kk-nb,     nb, c_zero, c_zero, dA(0,kk-nb),     ldda, queues[0] );
        //magmablas_claset( MagmaFull, m-(kk-nb), nb, c_zero, c_one,  dA(kk-nb,kk-nb), ldda, queues[0] );
    }

    if (kk > 0) {
        /* Use blocked code */
        for (i = ki; i >= nb; i -= nb) {
            ib = min(nb, k - i);
            /* Send current panel to the CPU for update */
            i__2 = m - i;
            //magma_cgetmatrix_async( i__2, ib, dA(i,i), ldda, panel, i__2, queues[0] );  // verify
            if (i + ib < n) {
                /* Apply H to A(i:m,i+ib:n) from the left */
                i__3 = n - i;

                magmablas_claset( MagmaFull, i,   ib, c_zero, c_zero, dA(0,i), ldda, queues[0] );
                magmablas_claset( MagmaFull, m-i, ib, c_zero, c_one,  dA(i,i), ldda, queues[0] );

                magma_clarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  i__2, i__3, ib,
                                  dA(i, i-nb), ldda, dT(i-nb),             ldt,
                                  dA(i, i), ldda, dwork, i__3, queues[0] );
            }

            /* Apply H to rows i:m of current block on the CPU */
            //magma_queue_sync( queues[0] );
            //lapackf77_cungqr(&i__2, &ib, &ib, panel, &i__2, &tau[i],
            //                 work, &lwork, &iinfo);
            //magma_csetmatrix_async( i__2, ib, panel, i__2, dA(i,i), ldda, queues[1] );  // verify

            /* Set rows 1:i-1 of current block to zero */
            i__2 = i + ib;
            //magmablas_claset( MagmaFull, i-ib,     ib, c_zero, c_zero, dA(0,i-ib),    ldda, queues[0] );
            //magmablas_claset( MagmaFull, m-(i-ib), ib, c_zero, c_one,  dA(i-ib,i-ib), ldda, queues[0] );
        }
    }

    magmablas_claset( MagmaFull, m, nb, c_zero, c_one, dA(0,0), ldda, queues[0] );

    magma_queue_sync( queues[0] );
    //magma_queue_sync( queues[1] );
    magma_queue_destroy( queues[0] );
    //magma_queue_destroy( queues[1] );
    magma_free( dwork );
    //magma_free_pinned( work );

    return *info;
} /* magma_cungqr_gpu */

#undef dA
#undef dT
