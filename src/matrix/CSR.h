#pragma once

#include <cusparse.h>

#include "matrix/vector.cu"
#include "../utility/setting.h"
#include "utility/utils.h"

#include <nsparse.h>

#define POWER_METHOD_MAX_ITER 100000

typedef struct{
  stype nnz; // number of non-zero
  stype n; // rows number
  stype m; // columns number

  bool on_the_device;
  bool is_symmetric;

  vtype *val; // array of nnz values
  itype *col; // array of the column index
  itype *row; // array of the pointer of the first nnz element of the rows

  // Matrix's cusparse descriptor
  cusparseMatDescr_t *descr;

}CSR;


namespace CSRm{
  int choose_mini_warp_size(CSR *A);
  CSR* init(stype n, stype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric);
  void partialAlloc(CSR *A, bool init_row, bool init_col, bool init_val);
  void free(CSR *A);
  void partialFree(CSR *A, bool val, bool col, bool row);
  void printInfo(CSR *A);
  void print(CSR *A, int type, int limit);
  bool equals(CSR *A, CSR *B);
  CSR* copyToDevice(CSR *A);
  CSR* copyToHost(CSR *A_d);
	CSR* clone(CSR *A);
  // matrix ops
  vector<vtype>* diag(CSR *A);
  CSR *T(cusparseHandle_t cusparse_h, CSR* A);
  CSR* CSRCSR_product(cusparseHandle_t handle, CSR *A, CSR *B, bool transA=false, bool transB=false);

  vector<vtype>* CSRVector_product_adaptive_miniwarp(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, vtype alpha=1., vtype beta=0.);
  vector<vtype>* CSRVector_product_CUSPARSE(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, bool trans=false, vtype alpha=1., vtype beta=0.);
  vector<vtype>* CSRVector_product_prolungator(CSR *A, vector<vtype> *x, vector<vtype> *y);

  vtype vectorANorm(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, CSR *A, vector<vtype> *x);
  //CSR* prune(cusparseHandle_t cusparse_h, CSR *A, const double threshold);
	vtype* toDense(cusparseHandle_t cusparse_h, CSR *A);
  void dismemberMatrix(cusparseHandle_t cusparse_h, CSR *A, CSR **U, CSR **L, vector<vtype> **D, cudaStream_t stream);
  vtype infinityNorm(CSR *A, cudaStream_t stream=DEFAULT_STREAM);
  void matrixVectorScaling(CSR *A, vector<vtype> *v, cudaStream_t stream=DEFAULT_STREAM);
  void iLU(cusparseHandle_t cusparse_h, CSR* A, bool trans);
  vector<vtype>* triangularSolve(cusparseHandle_t cusparse_h, cusparseSolveAnalysisInfo_t info, CSR *A, vector<vtype> *b, vector<vtype> *x, bool trans, vtype alpha);
  CSR* mergeDiagonal(CSR *A, vector<vtype> *D, cudaStream_t stream);
  vector<vtype>* LU_solve(cusolverSpHandle_t cusolver_h, CSR *A, vector<vtype> *b);
  vector<vtype>* absoluteRowSum(CSR *A, vector<vtype> *sum);

  double powerMethod(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, CSR *A, double toll, int maxiter=POWER_METHOD_MAX_ITER);


  //kernels
  template <int OP_TYPE>
  __global__ void _CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y);

}
