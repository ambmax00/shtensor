#ifndef LAPACK_H
#define LAPACK_H

extern "C" 
{
  void dgemm_(char* TRANSA,
		          char*	TRANSB,
              int* M,
              int* N,
              int* K,
              double* ALPHA,
              double*	A,
              int* LDA,
              double* B,
              int* LDB,
              double* BETA,
              double* C,
              int* LDC 
	);		

  void sgemm_(char* TRANSA,
		          char*	TRANSB,
              int* M,
              int* N,
              int* K,
              float* ALPHA,
              float*	A,
              int* LDA,
              float* B,
              int* LDB,
              float* BETA,
              float* C,
              int* LDC 
	);		
}

namespace LAPACK
{

static inline void dgemm(char _transa, char _transb, int _m, int _n, int _k, 
                         double _alpha, double* _a, int _lda, double* _b, int _ldb, 
                         double _beta, double* _c, int _ldc)
{
  dgemm_(&_transa,&_transb,&_m,&_n,&_k,&_alpha,_a,&_lda,_b,&_ldb,&_beta,_c,&_ldc);
}

static inline void sgemm(char _transa, char _transb, int _m, int _n, int _k, 
                         float _alpha, float* _a, int _lda, float* _b, int _ldb, 
                         float _beta, float* _c, int _ldc)
{
  sgemm_(&_transa,&_transb,&_m,&_n,&_k,&_alpha,_a,&_lda,_b,&_ldb,&_beta,_c,&_ldc);
}

} // end namespace LAPACK

#endif