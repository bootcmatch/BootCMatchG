//#pragma once

#include <iostream>
#include <curand.h>
#include "utility/setting.h"

#include "utility/utils.h"


#define FILL_KERNEL_BLOCKSIZE 1024

template <typename T>
struct vector{
  int n; // number of non-zero
  bool on_the_device;
  T *val; // array of nnz values
};

template <typename T>
struct vectorCollection{
  int n; // number of non-zero
  vector<T> **val;
};

namespace Vector{

  template <typename T>
  vector<T>* init(int n, bool allocate_mem, bool on_the_device);

  template <typename T>
  void fillWithValue(vector<T> *v, T value);

  template <typename T>
  void fillWithRandomValue(curandGenerator_t gen, vector<T> *v);/*{
    assert(v->on_the_device);
    curandGenerateUniformDouble(gen, v->val,  v->n);
  }
  */

  template <typename T>
  vector<T>* clone(vector<T> *a);

  template <typename T>
  vector<T>* copyToDevice(vector<T> *v);

  template <typename T>
  void copyTo(vector<T> *dest, vector<T> *source);

  template <typename T>
  vector<T>* copyToHost(vector<T> *v_d);

  template <typename T>
  void free(vector<T> *v);

  template <typename T>
  void print(vector<T> *v, int n_=-1);

  template <typename T>
  bool equals(vector<T> *a, vector<T> *b);

  template <typename T>
  int countNNV(vector<T> *v, T value);

  // only double for now:
  /*--------------------------------------------------------------------------
  * bcm_VectorInnerProd
  *--------------------------------------------------------------------------*/
  template <typename T>
  T dot(cublasHandle_t handle, vector<T> *a, vector<T> *b, int stride_a=1, int stride_b=1);

  template <typename T>
  void axpy(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, int inc=1);

  template <typename T>
  void scale(cublasHandle_t handle, vector<T> *x, T alpha, int inc=1);

  template <typename T>
  vector<T>* elementwise_div(vector<T> *a, vector<T> *b, vector<T> *c=NULL);

  template <typename T>
  T norm(cublasHandle_t handle, vector<T> *a, int stride_a=1);

template<typename T>
void printOnFile(vector<T> *v_, const char *path);

  template<typename T>
  void dump(vector<T> *v, const char *path);

  template<typename T>
  vector<T>* load(const char *path);

  // only for very hard DEBUG
  template<typename T>
  int checkEPSILON(vector<T> *A_);

  // vectorCollection of vector
  namespace Collection{

    template<typename T>
    vectorCollection<T>* init(int n);

    template<typename T>
    void free(vectorCollection<T> *c);
  }
}

namespace Vector{

  template <typename T>
  vector<T>* init(int n, bool allocate_mem, bool on_the_device){
    vector<T> *v = NULL;
    // on the host
    v = (vector<T>*) malloc( sizeof(vector<T>) );
    CHECK_HOST(v);

    v->n = n;
    v->on_the_device = on_the_device;

    if(allocate_mem){
      if(on_the_device){
        // on the device
        cudaError_t err;
        err = cudaMalloc( (void**) &v->val, n * sizeof(T) );
        CHECK_DEVICE(err);
      }else{
        // on the host
        v->val = (T*) malloc( n * sizeof(T) );
        CHECK_HOST(v->val);
      }
    }
    return v;
  }

  template<typename T>
  __global__ void _fillKernel(int n, T *v, const T val){
      int tidx = threadIdx.x + blockDim.x * blockIdx.x;
      int stride = blockDim.x * gridDim.x;

      for(; tidx < n; tidx += stride)
          v[tidx] = val;
  }

  template <typename T>
  void fillWithValue(vector<T> *v, T value){
    if(v->on_the_device){
      if(value == 0){
        cudaError_t err = cudaMemset(v->val, value, v->n * sizeof(T));
        CHECK_DEVICE(err);
      }else{
        dim3 block (FILL_KERNEL_BLOCKSIZE);
        dim3 grid ( ceil( (double) v->n / (double) block.x));
        _fillKernel<<<grid, block>>>(v->n, v->val, value);
      }
    }else{
      std::fill_n(v->val, v->n, value);
    }
  }


  template <typename T>
  void fillWithRandomValue(curandGenerator_t gen, vector<T> *v){
    assert(v->on_the_device);
    curandGenerateUniformDouble(gen, v->val,  v->n);
  }


  template <typename T>
  vector<T>* clone(vector<T> *a){

    assert( a->on_the_device );

    vector<T> *b = Vector::init<T>(a->n, true, true);

    cudaError_t err;
    err = cudaMemcpy(b->val, a->val, b->n * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_DEVICE(err);

    return b;
  }

  template <typename T>
  vector<T>* copyToDevice(vector<T> *v){

    assert( !v->on_the_device );

    int n = v->n;

    // alocate vector on the device memory
    vector<T> *v_d = init<T>(n, true, true);

    cudaError_t err = cudaMemcpy(v_d->val, v->val, n * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_DEVICE(err);

    return v_d;
  }

  #define copy_kernel_blocksize 1024
  template <typename T>
  __global__
  void _copy_kernel(itype n, T* dest, T* source){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

      dest[i] = source[i];
  }



  template <typename T>
  void copyTo(vector<T> *dest, vector<T> *source){

    //assert( dest->on_the_device && source->on_the_device && dest->n == source->n );

    int n = dest->n;

    //cudaError_t err = cudaMemcpy(dest->val, source->val, n * sizeof(T), cudaMemcpyDeviceToDevice);
    //CHECK_DEVICE(err);

    gridblock gb = gb1d(n, copy_kernel_blocksize);
     _copy_kernel<<<gb.g, gb.b>>>(n, dest->val, source->val);

  }

  template <typename T>
  vector<T>* copyToHost(vector<T> *v_d){

    assert( v_d->on_the_device );

    int n = v_d->n;

    // alocate vector on the host memory
    vector<T> *v = init<T>(n, true, false);

    cudaError_t err;

    err = cudaMemcpy(v->val, v_d->val, n * sizeof(T), cudaMemcpyDeviceToHost);
    CHECK_DEVICE(err);

    return v;
  }

  template <typename T>
  void free(vector<T> *v){
    if(v->on_the_device){
      cudaError_t err;
      err = cudaFree(v->val);
      CHECK_DEVICE(err);
    }else{
      std::free(v->val);
    }
    std::free(v);
  }

  template <typename T>
  void print(vector<T> *v, int n_){
    vector<T> *v_;

    int n;

    if(n_ == -1)
      n = v->n;
    else
      n = n_;

    if(v->on_the_device){
      v_ = Vector::copyToHost<T>(v);
    }else{
      v_ = v;
    }

    int i;
  	for(i=0; i<n; i++){
  		std::cout << v_->val[i];
      std::cout << " ";
  	}
  	std::cout << "\n\n";

    if(v->on_the_device){
      Vector::free<T>(v_);
    }

  }

  template <typename T>
  bool equals(vector<T> *a, vector<T> *b){

    assert(a->n == b->n);

    vector<T> *a_ = NULL;
    vector<T> *b_ = NULL;

    // implement GPU version, please
    if(a->on_the_device)
      a_ = Vector::copyToHost(a);
      else
    a_ = a;

    if(b->on_the_device)
      b_ = Vector::copyToHost(b);
    else
      b_ = b;

    for(int i=0; i<a->n; i++){
      if(a_->val[i] != b_->val[i]){
        std::cout << a_->val[i] << " " << b_->val[i];
        Vector::free(a_);
        Vector::free(b_);
        return false;
      }
    }
    return true;
  }

  template <typename T>
  int countNNV(vector<T> *v, T value){
    vector<T> *v_;

    if(v->on_the_device){
      std::cout << "WARN: vector copied on the host\n";
      v_ = Vector::copyToHost<T>(v);
    }else{
      v_ = v;
    }

    int i, c = 0;
    for(i=0; i<v_->n; i++){
      if(v_->val[i] != value){
        c++;
      }
    }

    if(v->on_the_device){
      Vector::free<T>(v_);
    }

    return c;
  }

  // only double for now:
  /*--------------------------------------------------------------------------
  * bcm_VectorInnerProd
  *--------------------------------------------------------------------------*/
  template <typename T>
  T dot(cublasHandle_t handle, vector<T> *a, vector<T> *b, int stride_a, int stride_b){

    assert(a->on_the_device == b->on_the_device);
    assert(a->n == b->n);

    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDdot(handle, a->n, a->val, stride_a, b->val, stride_b, &result);
    CHECK_CUBLAS(cublas_state);
    return result;
  }

  template <typename T>
  void axpy(cublasHandle_t handle, vector<T> *x, vector<T> *y, T alpha, int inc){

    assert(x->on_the_device == y->on_the_device);
    assert(x->n == y->n);

    cublasStatus_t cublas_state;
    cublas_state = cublasDaxpy(handle, x->n, &alpha, x->val, inc, y->val, inc);
    CHECK_CUBLAS(cublas_state);

  }

  template <typename T>
  void scale(cublasHandle_t handle, vector<T> *x, T alpha, int inc){

    assert(x->on_the_device);

    cublasStatus_t cublas_state;
    cublas_state = cublasDscal(handle, x->n, &alpha, x->val, inc);
    CHECK_CUBLAS(cublas_state);

  }

  #define elementwise_div_BLOCKSIZE 1024
  template <typename T>
  __global__
  void _elementwise_div(itype n, T *a, T *b, T *c){
    itype i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i >= n)
      return;

    c[i] = a[i] / b[i];
  }

  template <typename T>
  vector<T>* elementwise_div(vector<T> *a, vector<T> *b, vector<T> *c){
    assert(a->n == b->n);

    if(c == NULL)
      c = Vector::init<T>(a->n, true, true);

    gridblock gb = gb1d(a->n, elementwise_div_BLOCKSIZE);
    _elementwise_div<<<gb.g, gb.b>>>(a->n, a->val, b->val, c->val);

    return c;
  }

  template <typename T>
  T norm(cublasHandle_t handle, vector<T> *a, int stride_a){
    assert(a->on_the_device);

    //T result = sqrt( dot(handle, a, a) );
    T result;
    cublasStatus_t cublas_state;
    cublas_state = cublasDnrm2(handle, a->n, a->val, stride_a, &result);
    CHECK_CUBLAS(cublas_state);
    return result;
  }


/*
  template<typename T>
  void printOnFile(vector<T> *v_, const char *path){

    std::ofstream f;
    f.open (path);
    assert(f.is_open());

    vector<T> *v = NULL;
    if(v_->on_the_device)
      v = Vector::copyToHost(v_);
    else
      v = v_;

    for(int i=0; i<v->n; i++){
      f << v->val[i] << "\n";
    }

    if(v_->on_the_device)
      Vector::free(v);

    f.close();
  }
*/

template<typename T>
void printOnFile(vector<T> *v_, const char *path){

  FILE *file;
  file = fopen(path, "w");

  assert(file);

  vector<T> *v = NULL;
  if(v_->on_the_device)
    v = Vector::copyToHost(v_);
  else
    v = v_;

  for(int i=0; i<v->n; i++){
    fprintf(file, "%.14e\n", v->val[i]);
  }

  if(v_->on_the_device)
    Vector::free(v);

  fclose(file);
}

  template<typename T>
  void dump(vector<T> *v, const char *path){

    assert(!v->on_the_device);

    FILE *f = fopen(path, "wb");

    if(f == NULL){
      std::cerr << "File not found: " << path << "\n";
      exit(1);
    }

    fwrite(&v->n, sizeof(int) , 1, f);
    fwrite(v->val, sizeof(T) , v->n, f);

    fclose(f);
  }

  template<typename T>
  vector<T>* load(const char *path){

    FILE * f = fopen(path, "rb");
    if(f == NULL){
      std::cerr << "File not found: " << path << "\n";
      exit(1);
    }

    //for(int i=0; i<v->n; i++){
    int n = 0;
    fread(&n, sizeof(int) , 1, f);

    vector<T>* v = Vector::init<T>(n, true, false);
    fread(v->val, sizeof(T) , v->n, f);

    fclose(f);

    return v;
  }

  // only for very hard DEBUG
  template<typename T>
  int checkEPSILON(vector<T> *A_){
    vector<T> *A = NULL;

    if(A_->on_the_device)
      A = Vector::copyToHost(A_);
    else
      A = A_;

    int count = 0;
    for(int i=0; i<A->n; i++){

      if(A->val[i] == DBL_EPSILON)
        count++;
    }

    return count;
  }

  // vectorCollection of vector
  namespace Collection{

    template<typename T>
    vectorCollection<T>* init(int n){
      vectorCollection<T> *c = NULL;
      c = (vectorCollection<T>*) malloc(sizeof(vectorCollection<T>));
      CHECK_HOST(c);

      c->n = n;
      c->val = (vector<T>**) malloc( n * sizeof(vector<T>*));
      CHECK_HOST(c->val);

      for(int i=0; i<c->n; i++)
        c->val[i] = NULL;

      return c;
    }

    template<typename T>
    void free(vectorCollection<T> *c){

      for(int i=0; i<c->n; i++){
        if(c->val[i] != NULL)
          Vector::free(c->val[i]);
      }

      std::free(c->val);
      std::free(c);
    }
  }
}
