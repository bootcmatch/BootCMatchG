#pragma once

#include "FCG.h"

#define triple_innerproduct_blocksize 1024
__global__
void _triple_innerproduct(itype n, vtype *r, vtype *w, vtype *q, vtype *v, vtype *alpha_beta_gamma){
  __shared__ vtype alpha_shared[FULL_WARP];
  __shared__ vtype beta_shared[FULL_WARP];
  __shared__ vtype gamma_shared[FULL_WARP];

  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = threadIdx.x / FULL_WARP;

  int lane = tid % FULL_WARP;

  int i = tid;

  if(i >= n){
    if(lane == 0){
      alpha_shared[warp] = 0.;
      beta_shared[warp] = 0.;
      gamma_shared[warp] = 0.;
    }
    return;
  }

  vtype v_i = v[i];
  vtype alpha_i = r[i] * v_i;
  vtype beta_i = w[i] * v_i;
  vtype gamma_i = q[i] * v_i;

  #pragma unroll
  for(int k=FULL_WARP >> 1; k > 0; k = k >> 1){
    alpha_i += __shfl_down_sync(FULL_MASK, alpha_i, k);
    beta_i += __shfl_down_sync(FULL_MASK, beta_i, k);
    gamma_i += __shfl_down_sync(FULL_MASK, gamma_i, k);
  }

  if(lane == 0){
    alpha_shared[warp] = alpha_i;
    beta_shared[warp] = beta_i;
    gamma_shared[warp] = gamma_i;
  }

  __syncthreads();

  if(warp == 0){
    #pragma unroll
    for(int k=FULL_WARP >> 1; k > 0; k = k >> 1){
      alpha_shared[lane] += __shfl_down_sync(FULL_MASK, alpha_shared[lane], k);
      beta_shared[lane] += __shfl_down_sync(FULL_MASK, beta_shared[lane], k);
      gamma_shared[lane] += __shfl_down_sync(FULL_MASK, gamma_shared[lane], k);
    }

    if(lane == 0){
      atomicAdd(&alpha_beta_gamma[0], alpha_shared[0]);
      atomicAdd(&alpha_beta_gamma[1], beta_shared[0]);
      atomicAdd(&alpha_beta_gamma[2], gamma_shared[0]);
    }
  }
}

void triple_innerproduct(vector<vtype> *r, vector<vtype> *w, vector<vtype> *q, vector<vtype> *v, vtype *alpha, vtype *beta, vtype *gamma){

  assert(r->n == w->n && r->n == q->n && r->n == v->n);

  vector<vtype> *alpha_beta_gamma = Vector::init<vtype>(3, true, true);
  Vector::fillWithValue(alpha_beta_gamma, 0.);

  gridblock gb = gb1d(r->n, triple_innerproduct_blocksize);

  _triple_innerproduct<<<gb.g, gb.b>>>(r->n, r->val, w->val, q->val, v->val, alpha_beta_gamma->val);

  vector<vtype> *alpha_beta_gamma_host = Vector::copyToHost(alpha_beta_gamma);

  *alpha = alpha_beta_gamma_host->val[0];
  *beta = alpha_beta_gamma_host->val[1];
  *gamma = alpha_beta_gamma_host->val[2];

  Vector::free(alpha_beta_gamma);
}

//###########################################################################################################

#define double_merged_axpy_blocksize 1024
__global__
void _double_merged_axpy(itype n, vtype *x0, vtype *x1, vtype *x2, vtype alpha_0, vtype alpha_1){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i >= n)
    return;

  vtype xi1_local = alpha_0 * x0[i] + x1[i];
  x2[i] = alpha_1 * xi1_local + x2[i];
  x1[i] = xi1_local;
}


void double_merged_axpy(vector<vtype> *x0, vector<vtype> *x1, vector<vtype> *y, vtype alpha_0, vtype alpha_1){

  gridblock gb = gb1d(y->n, double_merged_axpy_blocksize);
  _double_merged_axpy<<<gb.g, gb.b>>>(y->n, x0->val, x1->val, y->val, alpha_0, alpha_1);

}

//###########################################################################################################


// bcm_PrecApply SRC\BOOTAMG\bcm_boot_prec.c
void preconditionApply(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vector<vtype> *rhs, vector<vtype> *x){

  vectorCollection<vtype> *RHS = FCG::context.RHS_buffer;
  vectorCollection<vtype> *Xtent = FCG::context.Xtent_buffer;

  if(bootamg_data->solver_type == 0){
    // multiplicative
    for(int k=0; k<boot_amg->n_hrc; k++){

      FCG::setHrrchBufferSize(boot_amg->H_array[k]);

      Vector::copyTo(RHS->val[0], rhs);
      Vector::copyTo(Xtent->val[0], x);

      GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, 1);

      Vector::copyTo(x, Xtent->val[0]);
    }
  }else if(bootamg_data->solver_type == 1){
     // symmetrized multiplicative
     for(int k=0; k<boot_amg->n_hrc; k++){

       FCG::setHrrchBufferSize(boot_amg->H_array[k]);

       Vector::copyTo(RHS->val[0], rhs);
       Vector::copyTo(Xtent->val[0], x);

       GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, 1);

       Vector::copyTo(x, Xtent->val[0]);
     }

     for(int k=boot_amg->n_hrc-1; k>=0; k--){

       FCG::setHrrchBufferSize(boot_amg->H_array[k]);

       /*
       int num_levels = boot_amg->H_array[k]->num_levels;
       for(int i=1; i<num_levels; i++){
         Vector::fillWithValue(RHS->val[i], 0.);
         Vector::fillWithValue(Xtent->val[i], 0.);
       }
       */

       Vector::copyTo(Xtent->val[0], x);
       Vector::copyTo(RHS->val[0], rhs);

       GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, 1);

       Vector::copyTo(x, Xtent->val[0]);

     }
  }else if(bootamg_data->solver_type == 2){
    // additive
    itype n = boot_amg->H_array[0]->A_array[0]->n;
    vector<vtype> *xadd = Vector::init<vtype>(n, true, true);
    Vector::fillWithValue(xadd, 0.);

    for(int k=0; k<boot_amg->n_hrc; k++){

      FCG::setHrrchBufferSize(boot_amg->H_array[k]);

      Vector::copyTo(Xtent->val[0], x);
      Vector::copyTo(RHS->val[0], rhs);

      GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, 1);

      Vector::axpy(h->cublas_h, Xtent->val[0], xadd, 1.);
    }

    vtype alpha = 1.0 / (vtype) boot_amg->n_hrc;
    Vector::scale(h->cublas_h, xadd, alpha);
    Vector::copyTo(x, xadd);
    Vector::free(xadd);
  }
}

vtype flexibileConjugateGradients(handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter){

  buildData *amg_data = bootamg_data->amg_data;
  CSR *A = amg_data->A;
  itype n = A->n;

  vector<vtype> *v = NULL;
  vector<vtype> *w = NULL;

  vectorCollection<vtype> *d = Vector::Collection::init<vtype>(2);
  d->val[0] = Vector::init<vtype>(n, true, true);
  Vector::fillWithValue(d->val[0], 0.);
  d->val[1] = Vector::init<vtype>(n, true, true);
  Vector::fillWithValue(d->val[1], 0.);

  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    v = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, x, NULL, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    v = CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, x, NULL, 1., 0.);
  #endif

  w = Vector::clone(rhs);
  Vector::axpy(h->cublas_h, v, w, -1.);

  vtype delta0 = Vector::norm(h->cublas_h, w);
  vtype rhs_norm = Vector::norm(h->cublas_h, rhs);

  if(delta0 <= DBL_EPSILON * rhs_norm){
    *num_iter = 0;
    exit(1);
  }

  if(precon){
    /* apply preconditioner to w */
    preconditionApply(h, bootamg_data, boot_amg, amg_cycle, w, d->val[0]);
  }

  vtype delta_old = Vector::dot(h->cublas_h, w, d->val[0]);

  if(delta_old <= 0.){
    std::cout << "\n ERROR1: indefinite preconditioner in cg_iter_coarse: " << delta_old << "\n";
    exit(-1);
  }

  int idx = 0, iter = 0;
  vtype l2_norm;

  do{
    #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
      CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, d->val[idx], v, false, 1., 0.);
    #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
      CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, d->val[idx], v, 1., 0.);
    #endif

    vtype tau = Vector::dot(h->cublas_h, d->val[idx], v);

    if (tau <= 0.){
      std::cout << "\n ERROR2: indefinite matrix in cg_iter_coarse: " << tau << "\n";
      exit(-2);
    }

    vtype alpha = delta_old / tau;

    // update solution
    Vector::axpy(h->cublas_h, d->val[idx], x, alpha);

    // update residual
    Vector::axpy(h->cublas_h, v, w, -alpha);

    l2_norm = Vector::norm(h->cublas_h, w);

    iter++;
    idx = iter % 2;

    Vector::fillWithValue(d->val[idx], 0.);

    if(precon){
      /* apply preconditioner to w */
      preconditionApply(h, bootamg_data, boot_amg, amg_cycle, w, d->val[idx]);
    }

    //update direction
    vtype tau1 = Vector::dot(h->cublas_h, d->val[idx], v);
    vtype beta = tau1 / tau;

    if(idx == 1)
      Vector::axpy(h->cublas_h, d->val[0], d->val[1], -beta);
    else
      Vector::axpy(h->cublas_h, d->val[1], d->val[0], -beta);

    delta_old = Vector::dot(h->cublas_h, w, d->val[idx]);

    if(VERBOSE > 0)
      std::cout << "bootpcg iteration: " << iter << "  residual: " << l2_norm << " relative residual: " << l2_norm / delta0 << "\n";

  }while(l2_norm > rtol * delta0 && iter < max_iter);

  assert( std::isfinite(l2_norm) );

  *num_iter = iter;

  if(precon){
    FCG::freePreconditionContext();
  }

  if(amg_cycle->cycle_type == 3)
    FCGK::freePreconditionContext();

  Vector::free(w);
  Vector::free(v);
  Vector::Collection::free(d);

  return l2_norm;
}

//###############################################################################################

vtype flexibileConjugateGradients_v2(handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter){

  buildData *amg_data = bootamg_data->amg_data;
  CSR *A = amg_data->A;
  itype n = A->n;

  vector<vtype> *v = Vector::init<vtype>(n, true, true);
  Vector::fillWithValue(v, 0.);
  vector<vtype> *w = NULL;
  vector<vtype> *r = NULL;
  vector<vtype> *d = NULL;
  vector<vtype> *q = NULL;

  r = Vector::clone(rhs);

  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    w = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, x, NULL, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    w = CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, x, NULL, 1., 0.);
  #endif

  Vector::axpy(h->cublas_h, w, r, -1.);

  vtype delta0 = Vector::norm(h->cublas_h, r);
  vtype rhs_norm = Vector::norm(h->cublas_h, rhs);

  if(delta0 <= DBL_EPSILON * rhs_norm){
    *num_iter = 0;
    exit(1);
  }

  if(precon){
    /* apply preconditioner to w */
    preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
  }

  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, v, w, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
  #endif

  vtype alpha = Vector::dot(h->cublas_h, r, v);
  vtype beta = Vector::dot(h->cublas_h, w, v);
  vtype delta = beta;
  vtype theta = alpha / delta;
  vtype gamma;

  // update solution
  Vector::axpy(h->cublas_h, v, x, theta);
  // update residual
  Vector::axpy(h->cublas_h, w, r, -theta);

  vtype l2_norm = Vector::norm(h->cublas_h, r);

  if (l2_norm <= rtol * delta0){
      *num_iter = 1;
  }

  int iter = 1;

  d = Vector::clone(v);
  q = Vector::clone(w);
  //d1 = Vector::init(n, true, true);
  //q1 = Vector::init(n, true, true);

  do{

    iter++;

    Vector::fillWithValue(v, 0.);

    if(precon){
      /* apply preconditioner to w */
      preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
    }

    #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
      CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, v, w, false, 1., 0.);
    #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
      CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
    #endif

    triple_innerproduct(r, w, q, v, &alpha, &beta, &gamma);
    theta = gamma / delta;

    delta = beta - pow(gamma, 2) / delta;
    vtype theta_2 = alpha / delta;

    Vector::axpy(h->cublas_h, d, v, -theta);
    Vector::copyTo(d, v);
    // update solution
    Vector::axpy(h->cublas_h, d, x, theta_2);

    Vector::axpy(h->cublas_h, q, w, -theta);
    Vector::copyTo(q, w);
    // update residual
    Vector::axpy(h->cublas_h, q, r, -theta_2);

    l2_norm = Vector::norm(h->cublas_h, r);

    if(VERBOSE > 0)
      std::cout << "bootpcg iteration: " << iter << "  residual: " << l2_norm << " relative residual: " << l2_norm / delta0 << "\n";

  }while(l2_norm > rtol * delta0 && iter < max_iter);

  assert( std::isfinite(l2_norm) );

  *num_iter = iter;

  if(precon){
    FCG::freePreconditionContext();
  }

  if(amg_cycle->cycle_type == 3)
    FCGK::freePreconditionContext();

  Vector::free(w);
  Vector::free(v);
  Vector::free(d);
  Vector::free(q);
  Vector::free(r);

  return l2_norm;
}

//###############################################################################################
vtype flexibileConjugateGradients_v3(handles *h, vector<vtype> *x, vector<vtype> *rhs, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int precon, int max_iter, double rtol, int *num_iter){

  buildData *amg_data = bootamg_data->amg_data;
  CSR *A = amg_data->A;
  itype n = A->n;

  vector<vtype> *v = Vector::init<vtype>(n, true, true);
  Vector::fillWithValue(v, 0.);
  vector<vtype> *w = NULL;
  vector<vtype> *r = NULL;
  vector<vtype> *d = NULL;
  vector<vtype> *q = NULL;

  r = Vector::clone(rhs);

  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    w = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, x, NULL, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    w = CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, x, NULL, 1., 0.);
  #endif

  Vector::axpy(h->cublas_h, w, r, -1.);

  vtype delta0 = Vector::norm(h->cublas_h, r);
  vtype rhs_norm = Vector::norm(h->cublas_h, rhs);

  if(delta0 <= DBL_EPSILON * rhs_norm){
    *num_iter = 0;
    exit(1);
  }

  if(precon){
    /* apply preconditioner to w */
    preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
  }

  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, v, w, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
  #endif

  vtype alpha = Vector::dot(h->cublas_h, r, v);
  vtype beta = Vector::dot(h->cublas_h, w, v);
  vtype delta = beta;
  vtype theta = alpha / delta;
  vtype gamma;

  // update solution
  Vector::axpy(h->cublas_h, v, x, theta);
  // update residual
  Vector::axpy(h->cublas_h, w, r, -theta);

  vtype l2_norm = Vector::norm(h->cublas_h, r);

  if (l2_norm <= rtol * delta0){
      *num_iter = 1;
  }

  int iter = 0;

  d = Vector::clone(v);
  q = Vector::clone(w);

  do{

    int idx = iter % 2;

    if(idx == 0){
      Vector::fillWithValue(v, 0.);

      if(precon){
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, v);
      }

      #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
        CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, v, w, false, 1., 0.);
      #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
        CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, v, w, 1., 0.);
      #endif

      triple_innerproduct(r, w, q, v, &alpha, &beta, &gamma);
    }else{
      Vector::fillWithValue(d, 0.);

      if(precon){
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, r, d);
      }

      #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
        CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, d, q, false, 1., 0.);
      #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
        CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, d, q, 1., 0.);
      #endif

      triple_innerproduct(r, q, w, d, &alpha, &beta, &gamma);
    }

    theta = gamma / delta;

    delta = beta - pow(gamma, 2) / delta;
    vtype theta_2 = alpha / delta;

    if(idx == 0){
      //Vector::axpy(h->cublas_h, d, v, -theta);
      // update solution
      //Vector::axpy(h->cublas_h, v, x, theta_2);
      double_merged_axpy(d, v, x, -theta, theta_2);

      //Vector::axpy(h->cublas_h, q, w, -theta);
      // update residual
      //Vector::axpy(h->cublas_h, w, r, -theta_2);
      double_merged_axpy(q, w, r, -theta, -theta_2);
    }else{
      //Vector::axpy(h->cublas_h, v, d, -theta);
      // update solution
      //Vector::axpy(h->cublas_h, d, x, theta_2);
      double_merged_axpy(v, d, x, -theta, theta_2);

      //Vector::axpy(h->cublas_h, w, q, -theta);
      // update residual
      //Vector::axpy(h->cublas_h, q, r, -theta_2);
      double_merged_axpy(w, q, r, -theta, -theta_2);
    }

    l2_norm = Vector::norm(h->cublas_h, r);

    if(VERBOSE > 0)
      std::cout << "bootpcg iteration: " << iter << "  residual: " << l2_norm << " relative residual: " << l2_norm / delta0 << "\n";

    iter++;

  }while(l2_norm > rtol * delta0 && iter < max_iter);

  assert( std::isfinite(l2_norm) );

  *num_iter = iter + 1;

  if(precon){
    FCG::freePreconditionContext();
  }

  if(amg_cycle->cycle_type == 3)
    FCGK::freePreconditionContext();

  Vector::free(w);
  Vector::free(v);
  Vector::free(d);
  Vector::free(q);
  Vector::free(r);

  return l2_norm;
}

//###############################################################################################


void preconditionApplyK(handles *h, int kk, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, int l, vector<vtype> *rhs, vector<vtype> *x){

  hierarchy *hrrch = boot_amg->H_array[kk];

  /*
  vectorCollection<vtype> *RHS = FCGK::context.RHS_buffer;
  vectorCollection<vtype> *Xtent = FCGK::context.Xtent_buffer;
  FCGK::setHrrchBufferSize(hrrch);
  Vector::fillWithValue(Xtent->val[l-1], 0.);
  */

  int num_levels = hrrch->num_levels;

  vectorCollection<vtype> *RHS = Vector::Collection::init<vtype>(num_levels);
  vectorCollection<vtype> *Xtent = Vector::Collection::init<vtype>(num_levels);

  // !skip the first
  for(int i=l-1; i<num_levels; i++){
    itype n_i = hrrch->A_array[i]->n;
    RHS->val[i] = Vector::init<vtype>(n_i, true, true);
    Vector::fillWithValue(RHS->val[i], 0.);
    Xtent->val[i] = Vector::init<vtype>(n_i, true, true);
    Vector::fillWithValue(Xtent->val[i], 0.);
  }

  Vector::copyTo(RHS->val[l-1], rhs);

  GAMG_cycle(h, kk, bootamg_data, boot_amg, amg_cycle, RHS, Xtent, l);

  Vector::copyTo(x, Xtent->val[l-1]);

  Vector::Collection::free(RHS);
  Vector::Collection::free(Xtent);

}

//###########################################################################


void inneritkcycle(handles *h, int kh, vector<vtype> *x, vector<vtype> *rhs, bootBuildData  *bootamg_data, boot * boot_amg, applyData *amg_cycle, double rtol, int l){

  hierarchy *hrrc = boot_amg->H_array[kh];

  CSR *A = hrrc->A_array[l];

  if(VERBOSE > 0)
    std::cout << "Start inneritkcyle level: " << l+1 << "\n";

  vectorCollection<vtype> *d = Vector::Collection::init<vtype>(2);
  d->val[0] = Vector::init<vtype>(A->n, true, true);
  Vector::fillWithValue(d->val[0], 0.);
  d->val[1] = Vector::init<vtype>(A->n, true, true);
  Vector::fillWithValue(d->val[1], 0.);

  vector<vtype> *w = Vector::clone(rhs);
  vtype delta0 = Vector::norm(h->cublas_h, w);

  if(VERBOSE > 0)
    std::cout << "Level " << l+1 << " delta0 " << delta0 << "\n";

  // apply preconditioner to w
  preconditionApplyK(h, kh, bootamg_data, boot_amg, amg_cycle, l+1, w, d->val[0]);

  vtype delta_old = Vector::dot(h->cublas_h, w, d->val[0]);

  if (VERBOSE > 1){
    vtype tnrm = Vector::norm(h->cublas_h, w);
    fprintf(stderr,"level %d recursion output W nrm   %g \n",l+1,tnrm);
    tnrm = Vector::norm(h->cublas_h, d->val[0]);
    fprintf(stderr,"level %d recursion output nrm   %g \n",l+1,tnrm);
  }

  if(delta_old <= 0.){
    std::cout << "\n ERROR1: indefinite preconditioner in inner_iter: " << delta_old << "\n";
    exit(-1);
  }

  // parse-matrix vector product
  #if CSR_VECTOR_MUL_GENERAL_TYPE == 0
    vector<vtype> *v = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, d->val[0], NULL, false, 1., 0.);
  #elif CSR_VECTOR_MUL_GENERAL_TYPE == 1
    vector<vtype> *v = CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, A, d->val[0], NULL, 1., 0.);
  #endif

  vtype tau = Vector::dot(h->cublas_h, d->val[0], v);

  if (tau <= 0.){
    std::cout << "\n ERROR2: indefinite matrix in inner_iter: " << tau << "\n";
    exit(-2);
  }

  vtype alpha = delta_old / tau;

  // update residual
  Vector::axpy(h->cublas_h, v, w, -alpha);

  vtype l2_norm = Vector::norm(h->cublas_h, w);

  if(VERBOSE > 0)
    fprintf(stderr,"level %d alpha %g l2_n %g rtol*delta0 %g \n", l+1, alpha, l2_norm, rtol * delta0);

  if(l2_norm <= rtol * delta0){
    // update solution
    Vector::axpy(h->cublas_h, d->val[0], x, alpha);
  }else{
    //apply preconditioner to w
    preconditionApplyK(h, kh, bootamg_data, boot_amg, amg_cycle, l+1, w, d->val[1]);
    vector<vtype> *v1 = CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, d->val[1], NULL, false, 1., 0.);

    vtype tau1, tau2, tau3, tau4;
    tau1 = Vector::dot(h->cublas_h, d->val[1], v); /* gamma of Notay algorithm */
    tau2 = Vector::dot(h->cublas_h, d->val[1], v1);/* beta of Notay algorithm */
    tau3 = Vector::dot(h->cublas_h, d->val[1], w); /* alpha 2 of Notay algorithm */
    tau4 = tau2 - pow(tau1, 2) / tau; /* rho2 of Notay algorihtm */

    if(VERBOSE > 0)
      fprintf(stderr,"tau 1:4 %g %g %g %g \n", tau1, tau2, tau3, tau4);

    // update solution
    alpha = alpha - (tau1 * tau3) / (tau * tau4);
    Vector::axpy(h->cublas_h, d->val[0], x, alpha);
    alpha = tau3 / tau4;
    Vector::axpy(h->cublas_h, d->val[1], x, alpha);
    Vector::free(v1);
  }

  if(VERBOSE > 0)
    fprintf(stderr,"End inneritkcyle level %d\n", l);

  Vector::free(v);
  Vector::free(w);
  Vector::Collection::free(d);

}
