#pragma once

#include <cusparse.h>
#include "matrix/CSR.h"

#include "matchingAggregation.h"
#include "unsymMatching.cu"

#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_reduce.cuh"

#include "suitor.cu"

#define MIN(a,b) (((a)<(b))?(a):(b))

namespace Matching{
  CSR* toMaximumProductMatrix(CSR *AH);
  CSR* makeAH(CSR *A, vector<vtype> *w);
  vector<itype>* suitor(CSR *A, vector<vtype> *w);
}

//####################################################################################
// ensure the numeric symmetry in the CSR matrix
__global__
void _write_T(itype n, vtype *val, itype *col, itype *row){
  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  for(int j=row[i]; j<row[i+1]; j++){
    itype c = col[j];

    if(i < c)
      break;

    vtype v = val[j];

    for(int jj=row[c]; jj<row[c+1]; jj++){
      if(col[jj] == i){
        val[jj] = v;
        break;
      }
    }
  }
}
//####################################################################################

#if TYPE_WRITE_T == 0

__forceinline__
__device__
int binsearch(int array[], unsigned int size, int value) {
  unsigned int low, high, medium;
  low=0;
  high=size;
  while(low<high) {
      medium=(high+low)/2;
      if(value > array[medium]) {
        low=medium+1;
      } else {
        high=medium;
      }
  }
  return low;
}



//################################################################################################
__global__
void _write_T_warp(itype n, int MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);
  vtype t;

  itype j_stop = A_row[warp+1];

  for(int j=A_row[warp]+lane; j<j_stop; j+=MINI_WARP_SIZE){
    itype c = A_col[j];

    if(warp < c)
       break;

    int nc = A_row[c+1] - A_row[c];

    int jj=binsearch(A_col+A_row[c], nc, warp);

    t=A_val[jj+A_row[c]];
    A_val[j]=t;
  }
}
//################################################################################################

#elif TYPE_WRITE_T == 1

__global__
void _write_T_warp(itype n, int MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype j_stop = A_row[warp+1];

  for(int j=A_row[warp]+lane; j<j_stop; j+=MINI_WARP_SIZE){
    itype c = A_col[j];

    if(warp < c)
      break;

    vtype v = A_val[j];

    for(int jj=A_row[c]; jj<A_row[c+1]; jj++){
      if(A_col[jj] == warp){
        A_val[jj] = v;
        break;
      }
    }
  }
}
#endif
//####################################################################################

// kernel che costruisce preventivamente il vettore C = d * w^2 usato in _makeAH
__global__ void _makeC(stype n, vtype *val, itype *col, itype *row, vtype *w, vtype *C){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

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
      C[r] = val[j] * pow(w[r], 2);
      break;
    }
  }
}

//####################################################################################

__global__ void _makeC_warp(stype n, int MINI_WARP_SIZE,  vtype *A_val, itype *A_col, itype *A_row, vtype *w, vtype *C){

  /*
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  itype j_start = A_row[warp];
  itype j_stop = A_row[warp+1];

  int j = j_start + lane, j_d;

  int j_d = WARP_SIZE, j;

  for(j = j_start+lane; ; j+=WARP_SIZE){
    int is_diag = __ballot_sync(warp_mask, ( (j < j_stop) && (A_col[j] == warp) ) ) ;
    j_d = __clz(is_diag);
    if(j_d != WARP_SIZE)
      break;
  }

  if(lane == 0)
    t_nnz_4r[warp+1] = j - j_start + (WARP_SIZE - j_d) - 1;
  */
}


__global__
void _makeAH_warp(stype n, int AH_MINI_WARP_SIZE, vtype *A_val, itype *A_col, itype *A_row, vtype *w, vtype *C, vtype *AH_val, itype *AH_col, itype *AH_row){

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  itype warp = tid / AH_MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % AH_MINI_WARP_SIZE;
  itype j_stop = A_row[warp+1];

  for(int j=A_row[warp]+lane; j<j_stop; j+=AH_MINI_WARP_SIZE){
    itype c = A_col[j];

    if(c != warp){
      vtype a = A_val[j];
      itype offset = c > warp ? warp + 1 : warp;
      AH_col[j - offset] = c;

      vtype norm = c > warp ? C[warp] + C[c] : C[c] + C[warp];
      if(norm > DBL_EPSILON){
        vtype w_temp = c > warp ? w[warp] * w[c] : w[c] * w[warp];
        AH_val[j - offset] = 1. - ( (2. * a * w_temp) / norm);
      }else
        AH_val[j - offset] = DBL_EPSILON;
    }
  }

  if(lane == 0){
    AH_row[warp+1] = j_stop - (warp + 1);
  }

  if(tid == 0){
    // set the first index of the row pointer to 0
    AH_row[0] = 0;
  }
}


//####################################################################################


//### original
__global__ void _makeAH(stype n, vtype *val, itype *col, itype *row, vtype *w, vtype *C, vtype *AH_val, itype *AH_col, itype *AH_row){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype r = i;
  itype j_start = row[i];
  itype j_stop = row[i+1];

  int j;

  // prima della diagonale, ogni elemento e' non zero
  int offset = r;
  for(j=j_start; j<j_stop; j++){
    itype c = col[j];
    vtype a = val[j];

    // abbiamo la diagonale nulla
    if(c == r){
      // metti l'offset a 1 in modo tale che i valori e colonne succesive vengano scritte in AH nella cella precedente
      offset += 1;
    }else{
      // salva colonna
      AH_col[j - offset] = c;

      vtype norm = c > r ? C[r] + C[c] : C[c] + C[r];
      if(norm > DBL_EPSILON){
        //AH_val[j - offset] = 1. - ( (2. * a * w[r] * w[c]) / norm);
        vtype w_temp = c > r ? w[r] * w[c] : w[c] * w[r];
        AH_val[j - offset] = 1. - ( (2. * a * w_temp) / norm);
      }else
        AH_val[j - offset] = DBL_EPSILON;
    }
  }
  // salva il fine riga
  AH_row[r+1] = j_stop - (r + 1);

  if(i == 0){
    // set the first index of the row pointer to 0
    AH_row[0] = 0;
  }

}


// funzione che presa in input la matrice CSR A, alloca e costruisce la rispettiva matrice AH
CSR* Matching::makeAH(CSR *A, vector<vtype> *w){

  assert(A->on_the_device);
  assert(w->on_the_device);

  stype n;
  n = A->n;

  // init a vector on the device
  vector<vtype> *C = Vector::init<vtype>(n, true, true);

  int miniwarp_size = CSRm::choose_mini_warp_size(A);

	gridblock gb = gb1d(n, makeC_BLOCKSIZE, false);
  _makeC<<<gb.g, gb.b>>>(n, A->val, A->col, A->row, w->val, C->val);

  CSR *AH = CSRm::init(A->n, A->m, (A->nnz - A->n), true, true, A->is_symmetric);

  gb = gb1d(n, makeAH_BLOCKSIZE, true, miniwarp_size);
  _makeAH_warp<<<gb.g, gb.b>>>(n, miniwarp_size, A->val, A->col, A->row, w->val, C->val, AH->val, AH->col, AH->row);

  Vector::free<vtype>(C);

  return AH;
}

// Binary operation for the CUB::Reduce in the find_Max_Min function
struct AbsMin
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &lhs, const T &rhs) const {
      T ab_lhs = fabs(lhs);
      T ab_rhs = fabs(rhs);
      return ab_lhs < ab_rhs  ? ab_lhs  : ab_rhs;
    }
};

//####################################################################################


// make the vector c needed in the _make_w kernel
__global__ void _make_c(stype n, vtype *val, itype *row, vtype *c, vtype *alpha_candidate){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype j_start = row[i];
  itype j_stop = row[i+1];

  vtype max = 0.;

  vtype min = DBL_MAX;
  vtype a;

  int j;
  for(j=j_start; j<j_stop; j++){
    a = log( fabs(val[j]) );
    if(a > max)
      max = a;
    if(a < min)
      min = a;
  }
  c[i] = max;
  alpha_candidate[i] = max - min;
}

//####################################################################################

#if MAXIMUM_PRODUCT_MATRIX_OP == 0
// Modify the values of the matrix A_HAT in order to transforms the objective from a maximum weight to maximum weight maximum cardinality
__global__ void _make_w(stype n, vtype *val, itype *col, itype *row, vtype *alpha, vtype *C){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype j_start = row[i];
  itype j_stop = row[i+1];

  vtype alpha_def = *alpha;

  int j;
  for(j=j_start; j<j_stop; j++){
    itype c = col[j];
    vtype a = val[j];
    val[j] = alpha_def + log( fabs(a) ) + (alpha_def - C[c]);
  }
}
#else
// Modify the values of the matrix A_HAT in order to transforms the objective from a maximum weight to maximum weight maximum cardinality
__global__ void _make_w(stype nnz, vtype *val, vtype *min){

  stype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= nnz)
    return;

  val[i] = log(  fabs(val[i]) / (0.999 * (*min))  );
}
#endif
//####################################################################################


// find the max (op_type==0) or the absolute min (op_type==1) in the input device array (with CUB utility)
vtype* find_Max_Min(vtype *a, stype n, int op_type){
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  vtype *min_max = NULL;
  cudaError_t err;
  err = cudaMalloc((void**)&min_max, sizeof(vtype) * 1);
  CHECK_DEVICE(err);

  if(op_type == 0){
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, min_max, n);
    // Allocate temporary storage
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, min_max, n);
  }else if(op_type == 1){
    AbsMin absmin;
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, a, min_max, n, absmin, DBL_MAX);
    // Allocate temporary storage
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, a, min_max, n, absmin, DBL_MAX);
  }

  err = cudaFree(d_temp_storage);
  CHECK_DEVICE(err);

  return min_max;
}

//####################################################################################


// Funzione che prende in input la matrice AH ne modifica i valore (inplace) per ottenere la matrice dei pesi W per il Maximum Product Matching
CSR* Matching::toMaximumProductMatrix(CSR *AH){
  assert(AH->on_the_device);

#if MAXIMUM_PRODUCT_MATRIX_OP == 0
  stype n;
  n = AH->n;
  vector<vtype> *c = Vector::init<vtype>(n, true, true);
  vector<vtype> *alpha_candidate = Vector::init<vtype>(n, true, true);

  gridblock gb = gb1d(n, make_c_BLOCKSIZE, false);

  _make_c<<<gb.g, gb.b>>>(n, AH->val, AH->row, c->val, alpha_candidate->val);

  // find alpha in alpha_candidate
  vtype *alpha = find_Max_Min(alpha_candidate->val, n, 0);
  Vector::free<vtype>(alpha_candidate);

  gb = gb1d(n, make_w_BLOCKSIZE, false);

  _make_w<<<gb.g, gb.b>>>(n, AH->val, AH->col, AH->row, alpha, c->val);

  Vector::free<vtype>(c);
  CHECK_DEVICE( cudaFree(alpha) );
#else
  // do W_data[j]=log(fabs(B_data[j])/(0.999*min_B));
  stype nnz = AH->nnz;
  // find the min value
  vtype *min = find_Max_Min(AH->val, nnz, 1);

  gridblock gb = gb1d(nnz, make_w_BLOCKSIZE, false);
  _make_w<<<gb.g, gb.b>>>(nnz, AH->val, min);
  CHECK_DEVICE( cudaFree(min) );
#endif

  return AH;
}

//####################################################################################

vector<itype>* Matching::suitor(CSR *A, vector<vtype> *w){

  assert(A->on_the_device && w->on_the_device);

#if MATCHING_MOD_TYPE == 1
  int miniwarp_size = CSRm::choose_mini_warp_size(A);
  gridblock gb = gb1d(A->n, write_T_BLOCKSIZE, true, miniwarp_size);
  _write_T_warp<<<gb.g, gb.b>>>(A->n, miniwarp_size, A->val, A->col, A->row);
#endif

  CSR *AH = Matching::makeAH(A, w);
  CSR *W = Matching::toMaximumProductMatrix(AH);

  vector<itype> *M = matchingAggregationContext::M_buffer;
  approx_match_gpu_suitor(W, M, matchingAggregationContext::ws_buffer, matchingAggregationContext::mutex_buffer);
  //std::cout << "CPU matching\n";
  //M = approx_match_cpu_suitor<vtype>(W);

#if MATCHING_MOD_TYPE == 0
  M = unsymFix(M);
#endif

  // W is an alias of AH
  CSRm::free(W);

  return M;
}
