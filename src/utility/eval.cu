#pragma once

#include <cusparse.h>
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_reduce.cuh"

#define MIN(a,b) (((a)<(b))?(a):(b))

typedef struct{
  vtype tot_w;
  int size;
} match_quality;

namespace Eval{

  void printMetaData(const char* name, double value, int type){
    printf("#META %s ", name);
    if(type == 0){
      int value_int = (int) value;
      printf("int %d", value_int);
    }
    else if(type == 1)
      printf("float %le", value);
    printf("\n");
  }

  struct CountNeg
  {
      template <typename T>
      __device__ __forceinline__
      T operator()(const T &lhs, const T &rhs) const {
        T ab_lhs = min(0, lhs);
        T ab_rhs = min(0, rhs);
        return ab_lhs + ab_rhs;
      }

  };

  __global__ void _removeSelfEdges(stype n, itype *M){
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

    if(i == M[i]){
      M[i] = -1;
    }

  }

  // assegna ad ogni coppia del matching M il rispettivo peso
  __global__ void _assignWtoM(stype n, itype *M, vtype *W_val, itype *W_col, itype *W_row, vtype *M_W){
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

    itype c = M[i];

    // no edge
    if(c == -1){
      M_W[i] = 0;
      return;
    }

    // only supT
    if(c < i+1){
      M_W[i] = 0;
      return;
    }

    itype j_start = W_row[i];
    itype j_stop = W_row[i+1];

    int j;
    for(j=j_start; j<j_stop; j++){
      itype c_j = W_col[j];

      if(c_j == c){
        vtype a = W_val[j];
        M_W[i] = a;
        return;
      }
    }
    // qualcosa e' andato storto (non esiste un arco (i, j) )
    M_W[i] = 0;
  }

  vtype getTotalMatchWeightMatching(vector<int> *M, CSR * W){

    void  *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    vector<vtype> *M_W = Vector::init<vtype>(M->n, true, true);

    dim3 block (assignWtoM_BLOCKSIZE);
    dim3 grid ( ceil( (double) M->n / (double) block.x));

    _assignWtoM<<<grid, block>>>(M->n, M->val, W->val, W->col, W->row, M_W->val);

    /*
    printf("\n");
    Vector::print(M);
    printf("\n");
    Vector::print(M_W);
    printf("\n");
    */

    vector<vtype> *tot_w = Vector::init<vtype>(1, true, true);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, M_W->val, tot_w->val, M_W->n);

    // Allocate temporary storage
    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, M_W->val, tot_w->val, M_W->n);
    err = cudaFree(d_temp_storage);
    CHECK_DEVICE(err);

    vector<vtype> *tot_w_host = Vector::copyToHost<vtype>(tot_w);

    Vector::free(M_W);
    Vector::free(tot_w);

    return *tot_w_host->val;
  }


  // calcola il peso totale (prodotto) del matching
  match_quality evaluateMatching(vector<int> *M, CSR * W){
    match_quality mq;

    // count matchin cardinality
    void  *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    dim3 block (assignWtoM_BLOCKSIZE);
    dim3 grid ( ceil( (double) M->n / (double) block.x));
    _removeSelfEdges<<<grid, block>>>(M->n, M->val);

    vector<int> *nnM = Vector::init<int>(1, true, true);

    CountNeg count_neg;
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, M->val, nnM->val, M->n, count_neg, 0);

    // Allocate temporary storage
    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, M->val, nnM->val, M->n, count_neg, 0);
    err = cudaFree(d_temp_storage);
    CHECK_DEVICE(err);

    vector<int> *nnM_host = Vector::copyToHost<int>(nnM);
    mq.size = M->n + *nnM_host->val;
    Vector::free(nnM);

    // tot weight calc
    mq.tot_w = getTotalMatchWeightMatching(M, W);

    return mq;
  }


  __global__ void _checkDuplicateEdge(stype n, itype *M, int *fail){
    stype i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

    itype a = M[i];

    for(int j=0; j<n; j++){
      if(i == j)
        continue;
      itype b = M[j];
      if(a == b){
        fail[0] = 0;
        return;
      }
    }
  }

  bool is_a_matching(vector<stype> *M, CSR *A){

    vtype tot_w = getTotalMatchWeightMatching(M, A);
    if(tot_w == 0.){
      // M non e' un matching
      return false;
    }

    vector<int> *fail = Vector::init<int>(1, true, true);
    Vector::fillWithValue(fail, 1);

    dim3 block (assignWtoM_BLOCKSIZE);
    dim3 grid ( ceil( (double) M->n / (double) block.x));

    _checkDuplicateEdge<<<grid, block>>>(M->n, M->val, fail->val);

    vector<int> *fail_host = Vector::copyToHost<int>(fail);
    Vector::free(fail);

    if(fail_host->val == 0){
      Vector::free<int>(fail_host);
      return false;
    }

    Vector::free<int>(fail_host);
    return true;
  }


  CSR* brutalCSR_sym(CSR *A_){
    CSR *A = CSRm::copyToHost(A_);

    for(int i=0; i<A->n; i++){
      for(int j=A->row[i]; j<A->row[i+1]; j++){

        int c = A->col[j];

        if(i < c)
          break;

        bool ok = false;

        for(int jj=A->row[c]; jj<A->row[c+1]; jj++){
          int cc = A->col[jj];
          if(cc == i){
            //std::cout << A->val[j] << " " << A->val[jj] << "\n";
            A->val[jj] = A->val[j];
            ok = true;

          }
        }
        assert(ok);

      }
    }

    CSRm::free(A_);
    A_ = CSRm::copyToDevice(A);
    CSRm::free(A);
    return A_;
  }

}
