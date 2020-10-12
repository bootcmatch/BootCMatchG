#include "matrix/CSR.h"

int CSRm::choose_mini_warp_size(CSR *A){

  int density = A->nnz / A->n;

  if(density < MINI_WARP_THRESHOLD_2)
    return 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    return 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    return 8;
  else if(density < MINI_WARP_THRESHOLD_16)
    return 16;
  else{
    return 32;
  }
}


CSR* CSRm::init(stype n, stype m, stype nnz, bool allocate_mem, bool on_the_device, bool is_symmetric){

  assert(n > 0 && m > 0 && nnz >= 0);

  CSR *A = NULL;

  // on the host
  A = (CSR*) malloc(sizeof(CSR));
  CHECK_HOST(A);

  A->nnz = nnz;
  A->n = n;
  A->m = m;

  A->on_the_device = on_the_device;
  A->is_symmetric = is_symmetric;

  if(allocate_mem){
    if(on_the_device){
      // on the device
      cudaError_t err;
      err = cudaMalloc( (void**) &A->val, nnz * sizeof(vtype) );
      CHECK_DEVICE(err);
      err = cudaMalloc( (void**) &A->col, nnz * sizeof(itype) );
      CHECK_DEVICE(err);
      err = cudaMalloc( (void**) &A->row, (n + 1) * sizeof(itype) );
      CHECK_DEVICE(err);
    }else{
      // on the host
      A->val = (vtype*) malloc( nnz * sizeof(vtype) );
      CHECK_HOST(A->val);
      A->col = (itype*) malloc( nnz * sizeof(itype) );
      CHECK_HOST(A->col);
      A->row = (itype*) malloc( (n + 1) * sizeof(itype) );
      CHECK_HOST(A->row);
    }
  }

  cusparseMatDescr_t *descr = NULL;
  descr = (cusparseMatDescr_t*) malloc( sizeof(cusparseMatDescr_t) );
  CHECK_HOST(descr);

  cusparseStatus_t  err = cusparseCreateMatDescr(descr);
  CHECK_CUSPARSE(err);

  cusparseSetMatIndexBase(*descr, CUSPARSE_INDEX_BASE_ZERO);

  if(is_symmetric)
    cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  else
    cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_GENERAL);

  A->descr = descr;
  return A;
}

void CSRm::partialAlloc(CSR *A, bool init_row, bool init_col, bool init_val){

  assert(A->on_the_device);

  cudaError_t err;
  if(init_val){
    err = cudaMalloc( (void**) &A->val, A->nnz * sizeof(vtype) );
    CHECK_DEVICE(err);
  }
  if(init_col){
    err = cudaMalloc( (void**) &A->col, A->nnz * sizeof(itype) );
    CHECK_DEVICE(err);
  }
  if(init_row){
    err = cudaMalloc( (void**) &A->row, (A->n + 1) * sizeof(itype) );
    CHECK_DEVICE(err);
  }
}

void CSRm::print(CSR *A, int type, int limit=0){
  CSR *A_ = NULL;

  if(A->on_the_device)
    A_ = CSRm::copyToHost(A);
  else
    A_ = A;

  switch(type) {
    case 0:
      printf("ROW:\n\t");
      if(limit == 0)
        limit = A_->n + 1;
      for(int i=0; i<limit; i++){
        printf("%d ", A_->row[i]);
      }
      break;
    case 1:
      printf("COL:\n\t");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        printf("%d ", A_->col[i]);
      }
      break;
    case 2:
      printf("VAL:\n\t");
      if(limit == 0)
        limit = A_->nnz;
      for(int i=0; i<limit; i++){
        printf("%lf ", A_->val[i]);
      }
      break;
  }
  printf("\n\n");

  if(A->on_the_device)
    CSRm::free(A_);
}

void CSRm::free(CSR *A){
  if(A->on_the_device){
    cudaError_t err;
    err = cudaFree(A->val);
    CHECK_DEVICE(err);
    err = cudaFree(A->col);
    CHECK_DEVICE(err);
    err = cudaFree(A->row);
    CHECK_DEVICE(err);
  }else{
    std::free(A->val);
    std::free(A->col);
    std::free(A->row);
  }
  CHECK_CUSPARSE( cusparseDestroyMatDescr(*A->descr) );
  std::free(A->descr);
  std::free(A);
}


void CSRm::partialFree(CSR *A, bool val, bool col, bool row){
  if(A->on_the_device){
    cudaError_t err;
    if(val){
      err = cudaFree(A->val);
      CHECK_DEVICE(err);
    }
    if(col){
      err = cudaFree(A->col);
      CHECK_DEVICE(err);
    }
    if(row){
      err = cudaFree(A->row);
      CHECK_DEVICE(err);
    }
  }else{
    if(val)
      std::free(A->val);
    if(col)
      std::free(A->col);
    if(row)
      std::free(A->row);
  }
  std::free(A->descr);
  std::free(A);
}

void CSRm::printInfo(CSR *A){
  printf("nnz: %d\n", A->nnz);
  printf("n: %d\n", A->n);
  printf("m: %d\n", A->m);
  if(A->is_symmetric)
    printf("SYMMETRIC\n");
  else
    printf("GENERAL\n");
	printf("\n");
}

CSR* CSRm::copyToDevice(CSR *A){

  assert( !A->on_the_device );

  stype n, m, nnz;

  n = A->n;
  m = A->m;

  nnz = A->nnz;

  // alocate CSR matrix on the device memory
  CSR *A_d = CSRm::init(n, m, nnz, true, true, A->is_symmetric);

  cudaError_t err;
  err = cudaMemcpy(A_d->val, A->val, nnz * sizeof(vtype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A_d->row, A->row, (n + 1) * sizeof(itype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A_d->col, A->col, nnz * sizeof(itype), cudaMemcpyHostToDevice);
  CHECK_DEVICE(err);

  return A_d;
}

CSR* CSRm::clone(CSR *A){

  assert( A->on_the_device );

  stype n, m, nnz;

  n = A->n;
  m = A->m;

  nnz = A->nnz;

  // alocate CSR matrix on the device memory
  CSR *B = CSRm::init(n, m, nnz, true, true, A->is_symmetric);

  cudaError_t err;

  err = cudaMemcpy(B->val, A->val, nnz * sizeof(vtype), cudaMemcpyDeviceToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(B->row, A->row, (n + 1) * sizeof(itype), cudaMemcpyDeviceToDevice);
  CHECK_DEVICE(err);
  err = cudaMemcpy(B->col, A->col, nnz * sizeof(itype), cudaMemcpyDeviceToDevice);
  CHECK_DEVICE(err);

  return B;
}

CSR* CSRm::copyToHost(CSR *A_d){

  assert( A_d->on_the_device );

  stype n, m, nnz;

  n = A_d->n;
  m = A_d->m;

  nnz = A_d->nnz;

  // alocate CSR matrix on the device memory
  CSR *A = CSRm::init(n, m, nnz, true, false, A_d->is_symmetric);

  cudaError_t err;

  err = cudaMemcpy(A->val, A_d->val, nnz * sizeof(vtype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->row, A_d->row, (n + 1) * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);
  err = cudaMemcpy(A->col, A_d->col, nnz * sizeof(itype), cudaMemcpyDeviceToHost);
  CHECK_DEVICE(err);

  return A;
}

// return a copy of A->T
CSR* CSRm::T(cusparseHandle_t cusparse_h, CSR* A){

  assert( A->on_the_device );

  cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
  cusparseIndexBase_t idxbase = CUSPARSE_INDEX_BASE_ZERO;

  CSR *AT = CSRm::init(A->m, A->n, A->nnz, true, true, A->is_symmetric);

  cusparseStatus_t err = cusparseDcsr2csc(cusparse_h, A->n, A->m, A->nnz, A->val, A->row, A->col, AT->val, AT->col, AT->row, copyValues, idxbase);
  CHECK_CUSPARSE(err);

  return AT;
}


__global__ void _getDiagonal_warp(itype n, int MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row, vtype *D){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype j_start = A_row[warp];
  itype j_stop = A_row[warp+1];

  int j_d = WARP_SIZE, j;

  for(j = j_start+lane; ; j+=MINI_WARP_SIZE){
    int is_diag = __ballot_sync(warp_mask, ( (j < j_stop) && (A_col[j] == warp) ) ) ;
    j_d = __clz(is_diag);
    if(j_d != MINI_WARP_SIZE)
      break;
  }

  //if(lane == 0)
    //D[warp] = j - j_start + (WARP_SIZE - j_d) - 1;
}


#define getDiagonal_BLOCKSIZE 1024
//SUPER temp kernel
__global__ void _getDiagonal(itype n, vtype *val, itype *col, itype *row, vtype *D){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype r = i;
  itype j_start = row[i];
  itype j_stop = row[i+1];

  int j;
  for(j=j_start; j<j_stop; j++){
    itype c = col[j];

    // if is a diagonal element
    if(c == r){
      D[i] = val[j];
    }
  }
}

// get a copy of the diagonal
vector<vtype>* CSRm::diag(CSR *A){
  vector<vtype> *D = Vector::init<vtype>(A->n, true, true);
  //Vector::fillWithValue(D, 0.)

  gridblock gb = gb1d(D->n, getDiagonal_BLOCKSIZE);
  _getDiagonal<<<gb.g, gb.b>>>(D->n, A->val, A->col, A->row, D->val);

  //int miniwarp_size = CSRm::choose_mini_warp_size(A);
  //gridblock gb = gb1d(D->n, getDiagonal_BLOCKSIZE, true, miniwarp_size);
  //_getDiagonal_warp<<<gb.g, gb.b>>>(D->n, miniwarp_size, A->val, A->col, A->row, D->val);

  return D;
}

/*--------------------------------------------------------------------------
 * bcm_VectorANorm:
 * Returns the A norm of the vector, where A is required to be CSR matrix.
 *--------------------------------------------------------------------------*/
vtype CSRm::vectorANorm(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, CSR *A, vector<vtype> *x){

  vector<vtype> *temp = CSRVector_product_CUSPARSE(cusparse_h, A, x, NULL, false, 1, 0);
  vtype norm = sqrt( Vector::dot(cublas_h, temp, x) );

  Vector::free(temp);

  return norm;
}

vtype* CSRm::toDense(cusparseHandle_t cusparse_h, CSR *A){

	assert( A->on_the_device );
	vtype * A_dense = NULL;
	cudaError_t err;
  err = cudaMalloc( (void**) &A_dense, (A->m * A->n) * sizeof(vtype) );
  CHECK_DEVICE(err);

	cusparseStatus_t err_spa = cusparseDcsr2dense(cusparse_h, A->m, A->n, *A->descr, A->val, A->row, A->col, A_dense, A->m);
	CHECK_CUSPARSE(err_spa);

	return A_dense;
}


#define row_sum_BLOCKSIZE 1024
__global__ void _row_sum(itype n, vtype *A_val, itype *A_row, vtype *sum){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  vtype local_sum = 0.;

  int j;
  for(j=A_row[i]; j<A_row[i+1]; j++)
    local_sum += fabs(A_val[j]);

    sum[i] = local_sum;
}

vtype CSRm::infinityNorm(CSR *A, cudaStream_t stream){
  assert(false);
  return 1.;
  /*
  assert(A->on_the_device);

  vector<vtype> *sum = Vector::init<vtype>(A->n, true, true);

  gridblock gb = gb1d(A->n, row_sum_BLOCKSIZE);
  _row_sum<<<gb.g, gb.b>>>(A->n, A->val, A->row, sum->val);

  vtype *norm = Parallel::max<vtype>(sum->val, sum->n, true, stream);
  Vector::free(sum);

  return *norm;*/
}

// MATRIX dismember function

#define dismemberPrepare_BLOCKSIZE 1024
__global__
void _dismemberPrepare(itype n, itype *A_col, itype *A_row, itype *t_nnz_4r){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % WARP_SIZE;

  itype j_start = A_row[warp];
  itype j_stop = A_row[warp+1];

  int j_d = WARP_SIZE, j;

  for(j = j_start+lane; ; j+=WARP_SIZE){
    unsigned active_lanes = __activemask();
    int is_diag = __ballot_sync(__activemask(), ( (j < j_stop) && (A_col[j] == warp) ) ) ;
    j_d = __clz(is_diag);
    if(j_d != WARP_SIZE)
      break;
  }

  if(lane == 0)
    t_nnz_4r[warp+1] = j - j_start + (WARP_SIZE - j_d) - 1;

  if(tid == 0)
    t_nnz_4r[0] = 0;
}


#define dismemberDo_BLOCKSIZE 1024
__global__ void _dismemberDo(itype n, vtype *A_val, itype *A_col, itype *A_row, vtype *L_val, itype *L_col, vtype *D, int *offset, int *t_nnz_4r){

  stype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % WARP_SIZE;

  itype j_start = A_row[warp];

  int offset_warp = offset[warp];
  int i = lane, j;

  for(j=j_start + lane; j<j_start+t_nnz_4r[warp+1]; j+=WARP_SIZE, i+=WARP_SIZE){
    L_col[offset_warp + i] = A_col[j];
    L_val[offset_warp + i] = A_val[j];
  }

  // fill the diagonal array
  if(lane == 0){
    D[warp] = A_val[ j_start + t_nnz_4r[warp+1] ];
  }
}


// get Upper, Lower triangle and Diagonal from the matrix A (A is symmetrix and its diagonal is not-empty)
void CSRm::dismemberMatrix(cusparseHandle_t cusparse_h, CSR *A, CSR **U, CSR **L, vector<vtype> **D, cudaStream_t stream=DEFAULT_STREAM){

  assert(false);
  /*
  assert( A->on_the_device );

  itype n = A->n, nnz = A->nnz;
  itype t_nnz = (nnz - n) / 2;

  *L = CSRm::init(n, n, t_nnz, false, true, A->is_symmetric);
  CSRm::partialAlloc(*L, false, true, true);

  *D = Vector::init<vtype>(n, true, true);

  vector<itype> *t_nnz_4r = Vector::init<itype>(n + 1, true, true);

  gridblock gb = gb1d(n, dismemberPrepare_BLOCKSIZE, true);
  _dismemberPrepare<<<gb.g, gb.b, 0, stream>>>(n, A->col, A->row, t_nnz_4r->val);

  itype *offset_val = Parallel::cumsum<itype>(t_nnz_4r->val, t_nnz_4r->n, false, stream);
  (*L)->row = offset_val;

  gb = gb1d(n, dismemberDo_BLOCKSIZE, true);
  _dismemberDo<<<gb.g, gb.b, 0, stream>>>(n, A->val, A->col, A->row, (*L)->val, (*L)->col, (*D)->val, offset_val, t_nnz_4r->val);

  Vector::free(t_nnz_4r);

  if( U != NULL){
    // U = L.T
    *U = CSRm::T(cusparse_h, *L);
  }
  */
}

//##################################################################################################
#define matrixVectorScaling_BLOCKSIZE 1024
__global__
void _matrixVectorScaling(itype n, vtype *A_val, itype *A_col, itype *A_row, vtype *v){

  stype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % WARP_SIZE;

  vtype scaler = v[warp];

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=WARP_SIZE)
    A_val[j] /= scaler;
}

void CSRm::matrixVectorScaling(CSR *A, vector<vtype> *v, cudaStream_t stream){

  assert(A->on_the_device && v->on_the_device);
  assert(A->n == v->n);

  gridblock gb = gb1d(A->n, matrixVectorScaling_BLOCKSIZE, true);
  _matrixVectorScaling<<<gb.g, gb.b, 0, stream>>>(A->n, A->val, A->col, A->row, v->val);
}
//##################################################################################################

void CSRm::iLU(cusparseHandle_t cusparse_h, CSR* A, bool trans=false){
  assert(A->on_the_device);

  cusparseSolveAnalysisInfo_t info;
  CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo(&info) );

  cusparseOperation_t op;
  if(trans){
    op = CUSPARSE_OPERATION_TRANSPOSE;
  }
  else{
    op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  }

  // symstem analysis
  CHECK_CUSPARSE( cusparseDcsrsv_analysis(cusparse_h, op, A->n, A->nnz, *A->descr, A->val, A->row, A->col, info) );

  // incoplete LU
  CHECK_CUSPARSE( cusparseDcsrilu0(cusparse_h, op, A->n, *A->descr, A->val, A->row, A->col, info) );

  cusparseDestroySolveAnalysisInfo(info);
}

vector<vtype>* CSRm::triangularSolve(cusparseHandle_t cusparse_h, cusparseSolveAnalysisInfo_t info, CSR *A, vector<vtype> *b, vector<vtype> *x, bool trans=false, vtype alpha=1.){


  cusparseOperation_t op;
  if(trans){
    op = CUSPARSE_OPERATION_TRANSPOSE;
  }
  else{
    op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  }

  if(x == NULL)
    x = Vector::init<vtype>(A->n, true, true);

  CHECK_CUSPARSE( cusparseDcsrsv_solve(cusparse_h, op, A->n, &alpha, *A->descr, A->val, A->row, A->col, info, b->val, x->val) );

  //cusparseDestroySolveAnalysisInfo(info);

  return x;
}


//##################################################################################################
#define mergeDiagonal_BLOCKSIZE 1024
__global__
void _mergeDiagonal(itype n, itype nnz, vtype *A_val, itype *A_col, itype *A_row, vtype *B_val, itype *B_col, itype *B_row, vtype *D){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % WARP_SIZE;
  itype last = A_row[warp+1];

  B_row[warp] = A_row[warp] + warp;

  for(int j=A_row[warp]+lane; j<last; j+=WARP_SIZE){
    B_col[j + warp] = A_col[j];
    B_val[j + warp] = A_val[j];
  }

  if(lane == 0){
    B_col[last + warp] = warp;
    B_val[last + warp] = D[warp];
  }

  if(tid == n - 1)
    B_row[n] = nnz;
}

CSR* CSRm::mergeDiagonal(CSR *A, vector<vtype> *D, cudaStream_t stream=DEFAULT_STREAM){
  assert(A->on_the_device && D->on_the_device);
  assert(A->n == D->n);

  itype n = A->n;
  CSR *B = CSRm::init(n, n, A->nnz + n, true, true, A->is_symmetric);

  gridblock gb = gb1d(n, mergeDiagonal_BLOCKSIZE, true);
  _mergeDiagonal<<<gb.g, gb.b, 0, stream>>>(n, B->nnz, A->val, A->col, A->row, B->val, B->col, B->row, D->val);

  return B;
}
//###################################################################################
#define absoluteRowSum_BLOCKSIZE 1024
#define row_sum_BLOCKSIZE 1024
__global__ void _row_sum_2(itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *sum){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  vtype local_sum = 0.;

  int j;
  for(j=A_row[i]; j<A_row[i+1]; j++)
      local_sum += fabs(A_val[j]);

    sum[i] = local_sum;
}

vector<vtype>* CSRm::absoluteRowSum(CSR *A, vector<vtype> *sum){
  assert(A->on_the_device);

  if(sum == NULL){
    sum = Vector::init<vtype>(A->n, true, true);
  }else{
    assert(sum->on_the_device);
  }

  //gridblock gb = gb1d(A->n, absoluteRowSum_BLOCKSIZE, true);
  gridblock gb = gb1d(A->n, absoluteRowSum_BLOCKSIZE, false);
  //_absoluteRowSum<<<gb.g, gb.b>>>(A->n, A->val, A->col, A->row, sum->val);
  //_row_sum<<<gb.g, gb.b>>>(A->n, A->val, A->row, sum->val);
  _row_sum_2<<<gb.g, gb.b>>>(A->n, A->val, A->row, A->col, sum->val);

  return sum;
}
//###################################################################################

/*
vector<vtype>* CSRm::LU_solve(cusolverSpHandle_t cusolver_h, CSR *A, vector<vtype> *b){

  assert(A->on_the_device && A->on_the_device);

  int singularity = 0;
  int reorder = 0;
  int tol = 1;
  vector<vtype> *x = Vector::init<vtype>(A->n, true, true);

  cusolverStatus_t err = cusolverSpDcsrlsvluHost(cusolver_h, A->n, A->nnz, *A->descr, A->val, A->row, A->col, b->val, tol, reorder, x->val, &singularity);
  CHECK_CUSOLVER(err);

  assert(singularity == -1);

  return x;
}

*/
//###################################################################################

template <typename T>
bool _arrayEqual(int n, T *a, T *b){
  for(int i=0; i<n; i++)
    if(a[i] != b[i])
      return false;
  return true;
}

// check if two CSR are equal ONLY FOR DEBUG PURPOSE
bool CSRm::equals(CSR *A, CSR *B){
  CSR *a_ = NULL;
  CSR *b_ = NULL;

  // implement GPU version, please
  if(A->on_the_device)
    a_ = CSRm::copyToHost(A);
  else
    a_ = A;

  if(B->on_the_device)
    b_ = CSRm::copyToHost(B);
  else
    b_ = B;

  bool result = _arrayEqual<itype>(a_->n+1, a_->row, b_->row) && _arrayEqual<itype>(a_->nnz, a_->col, b_->col) && _arrayEqual<vtype>(a_->nnz, a_->val, b_->val);
  return result;
}

// only for hard DEBUG
int check(CSR *A_){
  CSR *A = NULL;

  if(A_->on_the_device)
    A = CSRm::copyToHost(A_);
  else
    A = A_;

  int count = 0;
  for(int i=0; i<A->nnz; i++){

    if(!std::isfinite(A->val[i]) || A->val[i] == 0.)
      count++;
  }

  return count;
}

// only for hard DEBUG
int checkEPSILON(CSR *A_){
  CSR *A = NULL;

  if(A_->on_the_device)
    A = CSRm::copyToHost(A_);
  else
    A = A_;

  int count = 0;
  for(int i=0; i<A->nnz; i++){

    if(A->val[i] == DBL_EPSILON)
      count++;
  }

  return count;
}

//------------------------------------------------------------------------------

struct CSRInfo{
  int n;
  int m;
  int nnz;
  int max;
  int min;
  double sparsity;
  double variance;
};

namespace CSRmInfo{

  CSRInfo fetch(CSR *A_){

    CSR *A = NULL;

    if(A_->on_the_device)
      A = CSRm::copyToHost(A_);
    else
      A = A_;

    CSRInfo info;
    info.n = A->n;
    info.m = A->m;
    info.nnz = A->nnz;
    info.n = A->n;

    info.sparsity = (double) A->nnz / (double) A->n;
    info.variance = 0.;

    info.max = 0;
    info.min = info.nnz;

    for(int i=0; i<A->n; i++){
      int nnz_i = A->row[i+1] - A->row[i];
      info.variance += pow(info.sparsity - nnz_i, 2);

      info.max = nnz_i > info.max ? nnz_i : info.max;
      info.min = nnz_i < info.min ? nnz_i : info.min;
    }

    info.variance /= A->n;
    info.variance = sqrt(info.variance);


    if(A_->on_the_device)
      CSRm::free(A);

    return info;
  }

  void print(CSRInfo info){
    std::cout << info.n << " ";
    std::cout << info.m << " ";
    std::cout << info.nnz << " ";
    std::cout << info.sparsity << " ";
    std::cout << info.variance << " ";
    std::cout << info.min << " ";
    std::cout << info.max << " ";
    std::cout << "\n";
  }

  void printMeta(CSR **A_array, int n, const char *name){
    printf("#META agg;mtx_features_%s table ", name);
    for(int i=0; i<n; i++){
      CSR *Ai = A_array[i];
      CSRInfo info = CSRmInfo::fetch(Ai);

      std::cout << info.n << ";";
      std::cout << info.m << ";";
      std::cout << info.nnz << ";";
      std::cout << info.sparsity << ";";
      std::cout << info.variance << ";";
      std::cout << info.min << ";";
      std::cout << info.max << "";
      std::cout << "#";
    }
    std::cout << "\n";

  }

}

//------------------------------------------------------------------------------------------
//###########################################################################################
//------------------------------------------------------------------------------------------
//###########################################################################################
//------------------------------------------------------------------------------------------
//###########################################################################################

vector<vtype>* CSRm::CSRVector_product_CUSPARSE(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, bool trans, vtype alpha, vtype beta){
  stype n = A->n;


  stype y_n;
  stype m = A->m;

  cusparseOperation_t op;
  if(trans){
    op = CUSPARSE_OPERATION_TRANSPOSE;
    y_n = m;
    assert( x->n == n );
  }else{
    op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    y_n = n;
    assert( x->n == m );
  }

  if(y == NULL){
    assert( beta == 0. );
    y = Vector::init<vtype>(y_n, true, true);
  }

  cusparseStatus_t err = cusparseDcsrmv(cusparse_h, op, n, m, A->nnz, &alpha, *A->descr, A->val, A->row, A->col, x->val, &beta, y->val);
  CHECK_CUSPARSE(err);

  return y;
}

//------------------------------------------------------------------------------------------


#define CSR_vector_mul_mini_warp 1024
template <int OP_TYPE>
__global__
void CSRm::_CSR_vector_mul_mini_warp(itype n, int MINI_WARP_SIZE, vtype alpha, vtype beta, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    if(OP_TYPE == 0)
        T_i += (alpha * A_val[j]) * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 1)
        T_i += A_val[j] * __ldg(&x[A_col[j]]);
    else if(OP_TYPE == 2)
        T_i += -A_val[j] * __ldg(&x[A_col[j]]);
  }

  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
    if(OP_TYPE == 0)
        y[warp] = T_i + (beta * y[warp]);
    else if(OP_TYPE == 1)
        y[warp] = T_i;
    else if(OP_TYPE == 2)
      y[warp] = T_i + y[warp];
  }
}
//#########################################################################################


vector<vtype>* CSRm::CSRVector_product_adaptive_miniwarp(cusparseHandle_t cusparse_h, CSR *A, vector<vtype> *x, vector<vtype> *y, vtype alpha, vtype beta){
  stype n = A->n;

  int density = A->nnz / A->n;

  int min_w_size;

  if(density < MINI_WARP_THRESHOLD_2)
    min_w_size = 2;
  else if(density < MINI_WARP_THRESHOLD_4)
    min_w_size = 4;
  else if(density < MINI_WARP_THRESHOLD_8)
    min_w_size = 8;
  else if(density < MINI_WARP_THRESHOLD_16)
    min_w_size = 16;
  else{
    return CSRm::CSRVector_product_CUSPARSE(cusparse_h, A, x, y, false, alpha, beta);
  }

  if(y == NULL){
    assert( beta == 0. );
    y = Vector::init<vtype>(n, true, true);
  }

  gridblock gb = gb1d(n, CSR_vector_mul_mini_warp, true, min_w_size);

  if(alpha == 1. && beta == 0.){
    CSRm::_CSR_vector_mul_mini_warp<1><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }else if(alpha == -1. && beta == 1.){
    CSRm::_CSR_vector_mul_mini_warp<2><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }else{
    CSRm::_CSR_vector_mul_mini_warp<0><<<gb.g, gb.b>>>(n, min_w_size, alpha, beta, A->val, A->row, A->col, x->val, y->val);
  }
  return y;
}


//------------------------------------------------------------------------------------------------------

__global__
void _CSR_vector_mul_prolongator(itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *x, vtype *y){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= n)
    return;

  itype j = A_row[tid];
  y[tid] += A_val[j] * __ldg(&x[A_col[j]]);

}

vector<vtype>* CSRm::CSRVector_product_prolungator(CSR *A, vector<vtype> *x, vector<vtype> *y){
  stype n = A->n;

  assert( A->on_the_device );
  assert( x->on_the_device );

  gridblock gb = gb1d(n, CSR_vector_mul_mini_warp);

  _CSR_vector_mul_prolongator<<<gb.g, gb.b>>>(n, A->val, A->row, A->col, x->val, y->val);

  return y;
}



CSR* CSRCSR_product_cuSPARSE(cusparseHandle_t handle, CSR *A, CSR *B, bool transA=false, bool transB=false){

  assert(A->m == B->n);
  // i need to swap the name for the cusparse's lingo
  // number of rows of A
  itype m = A->n;
  // number of columns of A
  itype n = B->m;
  itype k = B->n;

  cusparseOperation_t opA, opB;
  if(transA){
    opA = CUSPARSE_OPERATION_TRANSPOSE;
  }
  else{
    opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  }

  if(transB){
    opB = CUSPARSE_OPERATION_TRANSPOSE;
  }
  else{
    opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  }

  // no memory allocation
  CSR *C = CSRm::init(m, n, 0, false, true, false);
  // allocate only row array
  CSRm::partialAlloc(C, true, false, false);

  //std::cout << m << " " << n << " " << k << " "  << A->nnz << " " << B->nnz << " ""\n";
  cusparseStatus_t state;
  // get nnz value for C; C->nnz is updated
  state = cusparseXcsrgemmNnz(handle, opA, opB, m, n, k, *A->descr, A->nnz, A->row, A->col, *B->descr, B->nnz, B->row, B->col, *C->descr, C->row, &C->nnz);
  CHECK_CUSPARSE(state);
  assert(&C->nnz != NULL);
  // allocate values and columns
  CSRm::partialAlloc(C, false, true, true);

  state = cusparseDcsrgemm(handle, opA, opB, m, n, k, *A->descr, A->nnz, A->val, A->row, A->col, *B->descr, B->nnz, B->val, B->row, B->col, *C->descr, C->val, C->row, C->col);
  CHECK_CUSPARSE(state);


  return C;
}

CSR* CSRCSR_product_nsparse(CSR *A, CSR *P){
  sfCSR mat_a, mat_p, mat_c;

  mat_a.M = A->n;
  mat_a.N = A->m;
  mat_a.nnz = A->nnz;

  mat_a.d_rpt = A->row;
  mat_a.d_col = A->col;
  mat_a.d_val = A->val;

  mat_p.M = P->n;
  mat_p.N = P->m;
  mat_p.nnz = P->nnz;

  mat_p.d_rpt = P->row;
  mat_p.d_col = P->col;
  mat_p.d_val = P->val;

  spgemm_kernel_hash(&mat_a, &mat_p, &mat_c);

  CSR* C = CSRm::init(mat_c.M, mat_c.N, mat_c.nnz, false, true, false);
  C->row = mat_c.d_rpt;
  C->col = mat_c.d_col;
  C->val = mat_c.d_val;

  //cudaDeviceSynchronize();
  return C;
}

CSR* CSRm::CSRCSR_product(cusparseHandle_t handle, CSR *A, CSR *B, bool transA, bool transB){
  CSR *C;
#if MATRIX_MATRIX_MUL_TYPE == 0
  // cusparse
  //C = CSRCSR_product_cuSPARSE(handle, A, B, transA, transB);
  C = CSRCSR_product_nsparse(A, B);
#elif MATRIX_MATRIX_MUL_TYPE == 1
  // nsparse
  printf("NSPARSE\n");
  C = CSRCSR_product_nsparse(A, B);
#endif

  return C;
}

double CSRm::powerMethod(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, CSR *A, double toll, int maxiter){

  vector<vtype> *z = Vector::init<vtype>(A->n, true, true);
  vector<vtype> *z_old = Vector::init<vtype>(A->n, true, true);
  Vector::fillWithValue(z_old, 1.);

  vtype l;
  vtype l_old;

  for(int i=0; i<maxiter; i++){
    CSRm::CSRVector_product_CUSPARSE(cusparse_h, A, z_old, z);
    l = Vector::norm(cublas_h, z);

    Vector::scale(cublas_h, z, 1 / l);

    if(i){
      vtype shift = (l - l_old) / l_old;
      if(shift < toll)
        break;
    }
    l_old = l;

    vector<vtype> *t = z_old;
    z_old = z;
    z = t;
  }

  Vector::free(z);
  Vector::free(z_old);

  return l;

}
