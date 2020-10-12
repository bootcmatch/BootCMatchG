#define PRG_SEED 1

struct handles{
  cudaStream_t stream1;
  cusparseHandle_t cusparse_h0, cusparse_h1;
  cublasHandle_t cublas_h;
  cusolverSpHandle_t cusolver_h;
  curandGenerator_t uniformRNG;
};

namespace Handles{
  handles* init(){

    handles *h = (handles*) malloc(sizeof(handles));
    CHECK_HOST(h);

    CHECK_CUSPARSE( cusparseCreate(&(h->cusparse_h0)) );
    CHECK_CUSPARSE( cusparseCreate(&(h->cusparse_h1)) );

    CHECK_CUBLAS( cublasCreate(&(h->cublas_h)) );

    CHECK_DEVICE( cudaStreamCreate(&(h->stream1)) );

    CHECK_CUSPARSE( cusparseSetStream(h->cusparse_h1, h->stream1) );

    CHECK_CUSOLVER( cusolverSpCreate(&(h->cusolver_h)) );

    curandCreateGenerator(&h->uniformRNG, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(h->uniformRNG, PRG_SEED);

    return h;
  }

  void free(handles *h){
    CHECK_CUSPARSE( cusparseDestroy(h->cusparse_h0) );
    CHECK_CUSPARSE( cusparseDestroy(h->cusparse_h1) );

    CHECK_CUBLAS( cublasDestroy(h->cublas_h) );

    CHECK_DEVICE( cudaStreamDestroy(h->stream1) );

    CHECK_CUSOLVER( cusolverSpDestroy(h->cusolver_h) );

    //curandDestroyGenerator(h->uniformRNG);

    std::free(h);
  }
}
