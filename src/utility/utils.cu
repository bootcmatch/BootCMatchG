#include "utility/utils.h"


gridblock gb1d(const unsigned n, const unsigned block_size, const bool is_warp_agg, int MINI_WARP_SIZE){
  gridblock gb;

  int n_ = n;

  if(is_warp_agg)
    n_ *= MINI_WARP_SIZE;

  dim3 block (block_size);
  dim3 grid ( ceil( (double) n_ / (double) block.x));

  gb.b = block;
  gb.g = grid;

  //printf("%d %d\n\n", gb.g.x, gb.b.x);

  return gb;
}


// cuSPARSE API errors
const char* cusparseGetStatusString(cusparseStatus_t error){
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:                  return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:          return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:             return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:         return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:           return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}



const char* cublasGetStatusString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

void CHECK_CUBLAS(cublasStatus_t err){
  const char *err_str = cublasGetStatusString(err);
  if(err != CUBLAS_STATUS_SUCCESS){
    printf("[ERROR CUBLAS] :\n\t%s\n", err_str);
    exit(1);
  }
}


//##############################################################################

namespace Parallel{
  template <typename T>
  T* max(T *a, int n, bool host_result, cudaStream_t stream){

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    T *max = NULL;
    cudaError_t err;
    err = cudaMalloc((void**)&max, sizeof(T));
    CHECK_DEVICE(err);

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, max, n, stream);
    // Allocate temporary storage
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, a, max, n, stream);

    err = cudaFree(d_temp_storage);
    CHECK_DEVICE(err);

    if(host_result){
      T *max_host = (T*) malloc(sizeof(T));
      CHECK_HOST(max_host);

      CHECK_DEVICE( cudaMemcpy(max_host, max, sizeof(T), cudaMemcpyDeviceToHost) );
      CHECK_DEVICE( cudaFree(max) );
      return max_host;
    }

    return max;
  }

  template <typename T>
  T* cumsum(T *a, int n, bool host_result, cudaStream_t stream){

    T *out = NULL;
    CHECK_DEVICE( cudaMalloc((void**)&out, n * sizeof(T)) );

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a, out, n);
    // Allocate temporary storage for inclusive prefix sum
    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CHECK_DEVICE(err);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a, out, n);

    CHECK_DEVICE( cudaFree(d_temp_storage) );


    if(host_result){
      T *out_host = (T*) malloc(sizeof(T));
      CHECK_HOST(out_host);

      CHECK_DEVICE( cudaMemcpy(out_host, out, n * sizeof(T), cudaMemcpyDeviceToHost) );
      CHECK_DEVICE( cudaFree(out) );
      return out_host;
    }

    return out;
  }
}

//##############################################################################

namespace TIME{

  int timer_index;
  int n;
  cudaEvent_t *starts, *stops;

  void init(){
    TIME::timer_index = 0;
    TIME::n = 0;
    TIME::starts = NULL;
    TIME::stops = NULL;
  }

  void addTimer(){
    TIME::starts = (cudaEvent_t*) realloc(TIME::starts, sizeof(cudaEvent_t) * TIME::n);
    CHECK_HOST(TIME::starts);
    TIME::stops = (cudaEvent_t*) realloc(TIME::stops, sizeof(cudaEvent_t) * TIME::n);
    CHECK_HOST(TIME::stops);
    cudaEventCreate(&TIME::starts[TIME::n-1]);
    cudaEventCreate(&TIME::stops[TIME::n-1]);
  }

  void start(){
    if(TIME::timer_index == TIME::n){
      TIME::n++;
      TIME::addTimer();
    }
    cudaEventRecord(TIME::starts[TIME::timer_index]);
    TIME::timer_index++;
  }

  float stop(){
    float milliseconds = 0.;
    cudaEvent_t start_ = TIME::starts[TIME::timer_index-1];
    cudaEvent_t stop_ = TIME::stops[TIME::timer_index-1];

    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    TIME::timer_index--;
    return milliseconds;
  }

  void free(){
    for(int i=0; i<TIME::n; i++){
      cudaEventDestroy( TIME::starts[i]);
      cudaEventDestroy( TIME::stops[i]);
    }
    std::free( TIME::starts);
    std::free( TIME::stops);
  }
}
