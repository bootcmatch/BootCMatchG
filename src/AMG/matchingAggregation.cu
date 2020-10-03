#pragma once

#include "matchingAggregation.h"
#include "cub/cub.cuh"
#include "relaxation.cu"

// kernels block size
#define make_P_BLOCKSIZE 1024
#define aggregate_BLOCKSIZE 1024

// algorithm contast
#define FTCOARSE_INC 100
#define COARSERATIO_THRSLD 1.2

double TOTAL_MUL_TIME;

//#############################################################################
void relaxPrepare(handles *h, int level, CSR *A, hierarchy *hrrch, buildData *amg_data, int force_relax_type=-1){

  int relax_type;

  if(force_relax_type != -1)
    relax_type = force_relax_type;
  else
    relax_type = amg_data->CRrelax_type;

  if(relax_type == 0){
    // jacobi
    if(hrrch->D_array[level] != NULL)
      Vector::free(hrrch->D_array[level]);
    hrrch->D_array[level] = CSRm::diag(A);

  }else if(relax_type == 1 || relax_type == 2){
    // Gauss Seidel
    assert(false);

  }else if(relax_type == 4){
    // L1 smoother
    if(hrrch->D_array[level] != NULL)
      Vector::free(hrrch->D_array[level]);
    hrrch->D_array[level] = CSRm::diag(A);

    if(hrrch->M_array[level] != NULL)
      Vector::free(hrrch->M_array[level]);
    hrrch->M_array[level] = CSRm::absoluteRowSum(A, NULL);

  }else if(relax_type == 5){
#if AFSAI == 1
    // afsai
    //  precond(A, GPUDevId, nProc, parPrec.nscal, parPrec.nstep, parPrec.stepSize, parPrec.epsilon);
    //int ITERATION_NUM = 6;
    //printf("\n\n aFSAI_PRE_ITER: %d\n\n", aFSAI_PRE_ITER);
    PRECOND *pre = precond2(A, aFSAI_PRE_ITER);
    hrrch->pre_array[level] = pre;
#else
    printf("ERROR] RELAX_TYPE NOT SUPPORTED\n");
    exit(1);
#endif
  }
}

//##################################################################################################
#define applyOmega_BLOCKSIZE 1024
__global__ void _applyOmega(itype n, vtype *A_val, itype *A_col, itype *A_row, const vtype omega){

  stype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % WARP_SIZE;

  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=WARP_SIZE){
    A_val[j] = -omega * A_val[j];
    if(A_col[j] == warp)
      A_val[j] += 1.;
  }
}

void applyOmega(CSR *A, const vtype omega, cudaStream_t stream=DEFAULT_STREAM){

  assert(A->on_the_device);

  gridblock gb = gb1d(A->n, applyOmega_BLOCKSIZE, true);
  _applyOmega<<<gb.g, gb.b, 0, stream>>>(A->n, A->val, A->col, A->row, omega);
}

//##################################################################################################

__global__
void _aggregate_symmetric(stype n, itype *A_row, vtype *P_val, itype *M, itype *markc, vtype *w, itype *nuns){

  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  itype v = i;
  itype u = M[i];

  // if it's a matched pair
  if(u != -1){

    vtype wv = w[v], wu = w[u];
    vtype normwagg = sqrt(wv * wv + wu * wu);

    if(normwagg > DBL_EPSILON){
      // good pair
      // if v is the pair master
      if(v < u){
        int nuns_local = atomicAdd(nuns, 1);
        markc[v] = nuns_local;
        markc[u] = nuns_local;

        P_val[v] = wv / normwagg;
        P_val[u] = wu / normwagg;
      }
      // if v,u is a good pair, exit
      return;
    }
  }

  // only single vertex and no-good pairs reach this point
  if( fabs(w[i]) > DBL_EPSILON ){
    // good single
    int nuns_local = atomicAdd(nuns, 1);
    markc[v] = nuns_local;
    P_val[v] = w[v] / fabs(w[v]);
  }else{
    // bad single
    int nuns_local = atomicAdd(nuns, 1);
    markc[v] = nuns_local;
    P_val[v] = 0.0;
  }
}


//####################################################################################

__global__
void _make_P_row(itype n, itype* P_row){

  itype v = blockDim.x * blockIdx.x + threadIdx.x;

  if(v > n)
    return;

  P_row[v] = v;
}

//####################################################################################


CSR* matchingPairAggregation(CSR *A, vector<vtype> *w){

  itype n = A->n;

  // Matching
  vector<itype> *M = Matching::suitor(A, w);

  gridblock gb;

  CSR *P = CSRm::init(n, 1, n, true, true, false);

  scalar<itype> *nuns = Scalar::init<itype>(0, true);

  gb = gb1d(n, aggregate_BLOCKSIZE);
  _aggregate_symmetric<<<gb.g, gb.b>>>(n, A->row, P->val, M->val, P->col, w->val, nuns->val);

  int* nuns_local = Scalar::getvalueFromDevice(nuns);
  Scalar::free(nuns);

  gb = gb1d(n, make_P_BLOCKSIZE);
  _make_P_row<<<gb.g, gb.b>>>(n, P->row);

  P->m = nuns_local[0];

  free(nuns_local);

  return P;
}

CSR* matchingAggregation(handles *h, buildData *amg_data, CSR *A, vector<vtype> **w, CSR **P, CSR **R){

  // A_{i-1}
  CSR *Ai_ = A, *Ai = NULL;
  CSR *Ri_ = NULL;
  // w_{i-1}
  vector<vtype> *wi_ = *w, *wi = NULL;

  double size_coarse, size_precoarse;
  double coarse_ratio;

  for(int i=0; i<amg_data->sweepnumber; i++){

    CSR *Pi_ = matchingPairAggregation(Ai_, wi_); /* routine with the real work. It calls the suitor procedure */
    //CSR *Pi_ = aggregateCPU(Ai_, wi_);

    // transpose
    Ri_ = CSRm::T(h->cusparse_h0, Pi_);

    TIME::start();

#if GALERKIN_PRODUCT_TYPE == 0
    // Pi-1.T * Ai-1
    CSR *temp = CSRm::CSRCSR_product(h->cusparse_h0, Ri_, Ai_, false, false);

    // Ai = (RA)P
    Ai = CSRm::CSRCSR_product(h->cusparse_h0, temp, Pi_, false, false);

#elif GALERKIN_PRODUCT_TYPE == 1
    // Pi-1.T * Ai-1
    CSR *temp = CSRm::CSRCSR_product(h->cusparse_h0, Ai_, Pi_, false, false);

    // Ai = R(AP)
    Ai = CSRm::CSRCSR_product(h->cusparse_h0, Ri_, temp, false, false);

#endif

    TOTAL_MUL_TIME += TIME::stop();

    CSRm::free(temp);

    //wi =  Pi-1.T * wi-1
    wi = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, Ri_, wi_, NULL, false);
    size_precoarse = Ai_->n;
    size_coarse = Ai->n;
    coarse_ratio = size_precoarse / size_coarse;

    if(i == 0)
      *P = Pi_;
    else{

      TIME::start();

      if(i == 1){
        //TODO special nsparse
        *P = CSRm::CSRCSR_product(h->cusparse_h0, *P, Pi_, false, false);
      }else{
        *P = CSRm::CSRCSR_product(h->cusparse_h0, *P, Pi_, false, false);
      }

      TOTAL_MUL_TIME += TIME::stop();

      CSRm::free(Ri_);
      Ri_ = NULL;
      CSRm::free(Pi_);
      CSRm::free(Ai_);
    }

    Vector::free(wi_);

    if (coarse_ratio <= COARSERATIO_THRSLD)
      amg_data->ftcoarse = FTCOARSE_INC;

    // exit condiction
    if(size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize)
      break;

    Ai_ = Ai;
    wi_ = wi;
  }

  *w = wi;
  if(Ri_ == NULL)
    *R = CSRm::T(h->cusparse_h0, *P);
  else
    *R = Ri_;

  return Ai;
}


hierarchy* adaptiveCoarsening(handles *h, buildData *amg_data){

  TOTAL_MUL_TIME = 0;

  CSR *A = amg_data->A;
  vector<vtype> *w = amg_data->w;

  vector<vtype> *w_temp = Vector::clone(w);

  CSR *P = NULL, *R = NULL;
  hierarchy *hrrch = AMG::Hierarchy::init(amg_data->maxlevels + 1);
  hrrch->A_array[0] = A;

  vtype normw = CSRm::vectorANorm(h->cusparse_h0, h->cublas_h, A, w_temp);

  vtype avcoarseratio = 0.;
  int level = 0;
  relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);

  matchingAggregationContext::initContext(A->n);

  amg_data->ftcoarse = 1;

  TIME::start();

  if(normw > DBL_EPSILON){
    for(level=1; level < amg_data->maxlevels;){

      hrrch->A_array[level] = matchingAggregation(h, amg_data, hrrch->A_array[level-1], &w_temp, &P, &R);
      //CSRMatrixPrintMM(hrrch->A_array[level], "/home/pasquini/singularity.mtx");

      if(!amg_data->agg_interp_type){
        // #change STREAM
        relaxPrepare(h, level, hrrch->A_array[level], hrrch, amg_data);
      }

      hrrch->P_array[level-1] = P;
      hrrch->R_array[level-1] = R;

      vtype size_coarse = hrrch->A_array[level]->n;
      vtype coarse_ratio = hrrch->A_array[level-1]->n / size_coarse;
      avcoarseratio = avcoarseratio + coarse_ratio;

      level++;
      // exit condiction
      if(size_coarse <= amg_data->ftcoarse * amg_data->maxcoarsesize)
        break;
    }
  }else{
    std::cout << "Warning: no need to build multigrid since the matrix is well conditioned\n";
  }

  float aggregation_time = TIME::stop();

// ##############################################################################################
  if(amg_data->agg_interp_type == 1){

    for(int j=0; j<level-1; j++){

      CSR *A = hrrch->A_array[j];

      vector<vtype> *D = hrrch->D_array[j];
      //vector<vtype> *D = CSRm::diag(A);
      assert(D != NULL);

      CSR *A_temp = CSRm::clone(A);

      CSRm::matrixVectorScaling(A_temp, D);

      vtype omega = 4.0 / ( 3.0 * CSRm::infinityNorm(A_temp) );

      applyOmega(A_temp, omega);

      CSR *P_temp = hrrch->P_array[j];
	    hrrch->P_array[j] = CSRm::CSRCSR_product(h->cusparse_h0, A_temp, P_temp);

      CSRm::free(P_temp);
      CSRm::free(A_temp);

      // transpose
      hrrch->R_array[j] = CSRm::T(h->cusparse_h0, hrrch->P_array[j]);

      A_temp = CSRm::CSRCSR_product(h->cusparse_h0, hrrch->R_array[j], hrrch->A_array[j]);

      CSRm::free(hrrch->A_array[j+1]);

      hrrch->A_array[j+1] = CSRm::CSRCSR_product(h->cusparse_h0, A_temp, hrrch->P_array[j]);

      relaxPrepare(h, j+1, hrrch->A_array[j+1], hrrch, amg_data);

      CSRm::free(A_temp);
    }
  }
// ##############################################################################################

  AMG::Hierarchy::finalize_level(hrrch, level);

  if(amg_data->coarse_solver == 9){
    LU_factor *luf = LU::setup(h->cusparse_h0, hrrch->A_array[level - 1]);
    hrrch->LU = luf;
  }else{
    // in order to apply, to the coarsest matrix, the correct relax-preprocessing
    if(amg_data->coarse_solver != amg_data->CRrelax_type ){
      relaxPrepare(h, level-1, hrrch->A_array[level-1], hrrch, amg_data, amg_data->coarse_solver);
    }
  }

  AMG::Hierarchy::finalize_cmplx(hrrch);
  AMG::Hierarchy::finalize_wcmplx(hrrch);
  hrrch->avg_cratio = avcoarseratio / (level-1);

  AMG::Hierarchy::printInfo(hrrch);

  //Eval::printMetaData("time;aggregation_time", aggregation_time, 1);

  Eval::printMetaData("agg;level_number", level-1, 0);
  Eval::printMetaData("agg;avg_coarse_ratio", hrrch->avg_cratio, 1);
  Eval::printMetaData("agg;OpCmplx", hrrch->op_cmplx, 1);
  //Eval::printMetaData("agg;OpCmplxW", hrrch->op_wcmplx, 1);
  //Eval::printMetaData("agg;coarsest_size", hrrch->A_array[level-1]->n, 0);

  Vector::free(w_temp);
  matchingAggregationContext::freeContext();

  std::cout <<  "TOTAL_MUL_TIME: " << TOTAL_MUL_TIME << "\n\n";
  return hrrch;
}


//#############################################################################
