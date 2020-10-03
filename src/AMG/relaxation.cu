#pragma once

struct relaxContext{
  vector<vtype> *temp_buffer;
  itype n;
};

namespace Relax{
  relaxContext context;

  void initContext(itype n){
    Relax::context.temp_buffer = Vector::init<vtype>(n, true, true);
    Relax::context.n = n;
  }

  void set_n_context(itype n){
    Relax::context.temp_buffer->n = n;
  }

  void freeContext(){
    Relax::set_n_context(Relax::context.n);
    Vector::free(Relax::context.temp_buffer);
  }
}

//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
#define jacobi_BLOCKSIZE 1024

__global__
void _aFSAI_FT_apply(int mw_size, itype n, vtype *FT_val, itype *FT_row, itype *FT_col, vtype *u, vtype *Fr, vtype omega){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / mw_size;

  if(warp >= n)
    return;

  int lane = tid % mw_size;
  int mask_id = (tid % FULL_WARP) / mw_size;
  int warp_mask = getMaskByWarpID(mw_size, mask_id);

  vtype T_i = 0.;

  // F * r
  for(int j=FT_row[warp]+lane; j<FT_row[warp+1]; j+=mw_size){
    T_i += FT_val[j] * __ldg(&Fr[FT_col[j]]);
  }
  for(int k=mw_size >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0)
      u[warp] = u[warp] + omega * T_i;

}


// r = f - Au
__global__
void _compute_residue(int mw_size, itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *u, vtype *f, vtype *residue){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / mw_size;

  if(warp >= n)
    return;

  int lane = tid % mw_size;
  int mask_id = (tid % FULL_WARP) / mw_size;
  int warp_mask = getMaskByWarpID(mw_size, mask_id);

  vtype T_i = 0.;

  // A * u
  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=mw_size){
    T_i += A_val[j] * __ldg(&u[A_col[j]]);
  }

  // WARP sum reduction
  for(int k=mw_size >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0)
      residue[warp] = f[warp] - T_i;
}

#if AFSAI == 1
vector<vtype>* aFSAI_smoother(handles *h, CSR *A, PRECOND *pre, vector<vtype> *f, vector<vtype> *u, vector<vtype> *temp, bool forward, int k){
  double omega = 1.;//pre->omega;

  CSR *F = pre->Pre;
  CSR *FT = pre->PreT;

  Relax::set_n_context(A->n);
  vector<vtype> *w = Relax::context.temp_buffer;

  for(int i=0; i<k; i++){
     int mw_size = CSRm::choose_mini_warp_size(A);
     gridblock gb = gb1d(A->n, jacobi_BLOCKSIZE, true, mw_size);
     _compute_residue<<<gb.g, gb.b>>>(mw_size, A->n, A->val, A->row, A->col, u->val, f->val, w->val);

     CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, F, w, temp);

     mw_size = CSRm::choose_mini_warp_size(FT);
     gb = gb1d(A->n, jacobi_BLOCKSIZE, true, mw_size);
     _aFSAI_FT_apply<<<gb.g, gb.b>>>(mw_size, FT->n, FT->val, FT->row, FT->col, u->val, temp->val, omega);
  }

  return u;
}


vector<vtype>* naive_aFSAI_smoother(handles *h, CSR *A, PRECOND *pre, vector<vtype> *f, vector<vtype> *u, bool forward, int k){

  double omega = 1.;//pre->omega;

  CSR *F = pre->Pre;
  CSR *FT = pre->PreT;

  vector<vtype> *utemp = Vector::init<vtype>(F->n, true, true);
  vector<vtype> *temp = Vector::init<vtype>(F->n, true, true);

  for(int i=0; i<k; i++){
    //Relax::set_n_context(n);
    //vector<vtype> *w = Relax::context.temp_buffer;

    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, A, u, utemp);
    vector<vtype> *w = Vector::clone(f);
    Vector::axpy(h->cublas_h, utemp, w, -1.);
    //Vector::free(utemp);
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, F, w, utemp);
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, FT, utemp, temp);

    Vector::axpy(h->cublas_h, temp, u, omega);
    Vector::free(w);
  }

  Vector::free(utemp);
  Vector::free(temp);


  return u;
}
//°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
#endif


#define jacobi_hybrid_BLOCKSIZE 1024
__global__
void _jacobi_hybrid(itype n, vtype relax_weight, vtype *T, vtype *D, vtype *u, vtype *f, vtype *u_){
  itype i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
    u_[i] = ( (-T[i] + f[i]) / D[i] ) + u[i];
}


__inline__
vector<vtype>** jacobi_cusparse_hybrid(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight){
  itype n = A->n;

  Relax::set_n_context(n);
  vector<vtype> *T = Relax::context.temp_buffer;

  CSRm::CSRVector_product_CUSPARSE(cusparse_h, A, u, T);

  gridblock gb = gb1d(n, jacobi_hybrid_BLOCKSIZE);
  _jacobi_hybrid<<<gb.g, gb.b>>>(n, relax_weight, T->val, D->val, u->val, f->val, (*u_)->val);

  return u_;
}


#define jacobi_BLOCKSIZE 1024
template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _jacobi_it(itype n, vtype relax_weight, vtype *A_val, itype *A_row, itype *A_col, vtype *D, vtype *u, vtype *f, vtype *u_){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  // A * u
  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    T_i += A_val[j] * __ldg(&u[A_col[j]]);
  }

  // WARP sum reduction
  #pragma unroll MINI_WARP_SIZE
  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0){
    if(OP_TYPE == 0)
      u_[warp] = ( (-T_i + f[warp]) / D[warp] ) + u[warp];
    else if(OP_TYPE == 1)
      u_[warp] = ( -T_i / D[warp] ) + u[warp];
  }
}

vector<vtype>* jacobi_adaptive_miniwarp(cusparseHandle_t cusparse_h, cublasHandle_t cublas_h, int k, CSR *A, vector<vtype> *u, vector<vtype> **u_, vector<vtype> *f, vector<vtype> *D, vtype relax_weight){

  assert(f != NULL);
  vector<vtype> *swap_temp;

#if CSR_JACOBI_TYPE == 1

  for(int i=0; i<k; i++){
    swap_temp = u;
    u = *jacobi_cusparse_hybrid(cusparse_h, cublas_h, k, A, u, u_, f, D, relax_weight);
    *u_ = swap_temp;
  }

#elif CSR_JACOBI_TYPE == 0

  itype n = A->n;
  int density = A->nnz / A->n;

  if(density <  MINI_WARP_THRESHOLD_2){
    //miniwarp 2
    gridblock gb = gb1d(n, jacobi_BLOCKSIZE, true, 2);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 2><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if (density <  MINI_WARP_THRESHOLD_4){
    //miniwarp 4
    gridblock gb = gb1d(n, jacobi_BLOCKSIZE, true, 4);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 4><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if (density <  MINI_WARP_THRESHOLD_8){
    //miniwarp 8
    gridblock gb = gb1d(n, jacobi_BLOCKSIZE, true, 8);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 8><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else if(density < MINI_WARP_THRESHOLD_16){
    //miniwarp 16
    gridblock gb = gb1d(n, jacobi_BLOCKSIZE, true, 16);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 16><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }else{
    //miniwarp 32
    gridblock gb = gb1d(n, jacobi_BLOCKSIZE, true, 32);

    for(int i=0; i<k; i++){
      _jacobi_it<0, 32><<<gb.g, gb.b>>>(n, relax_weight, A->val, A->row, A->col, D->val, u->val, f->val, (*u_)->val);
      swap_temp = u;
      u = *u_;
      *u_ = swap_temp;
    }
  }

#endif


  return u;
}

#define jacobi_BLOCKSIZE 1024
template <int OP_TYPE, int MINI_WARP_SIZE>
__global__
void _compute_residue(itype n, vtype *A_val, itype *A_row, itype *A_col, vtype *u, vtype *u_){
  itype tid = blockDim.x * blockIdx.x + threadIdx.x;

  int warp = tid / MINI_WARP_SIZE;

  if(warp >= n)
    return;

  int lane = tid % MINI_WARP_SIZE;
  int mask_id = (tid % FULL_WARP) / MINI_WARP_SIZE;
  int warp_mask = getMaskByWarpID(MINI_WARP_SIZE, mask_id);

  vtype T_i = 0.;

  // A * u
  for(int j=A_row[warp]+lane; j<A_row[warp+1]; j+=MINI_WARP_SIZE){
    T_i += A_val[j] * __ldg(&u[A_col[j]]);
  }

  // WARP sum reduction
  #pragma unroll MINI_WARP_SIZE
  for(int k=MINI_WARP_SIZE >> 1; k > 0; k = k >> 1){
    T_i += __shfl_down_sync(warp_mask, T_i, k);
  }

  if(lane == 0)
      u_[warp] = u[warp] - T_i;

}


/*
vector<vtype>* aFSAI_smoother(handles *h, CSR *A, PRECOND *pre, vector<vtype> *rhs, vector<vtype> *x_tent, bool forward){

  double a = 1., b = 0.0, omega = pre->omega;

  CSR *Pre = pre->Pre;
  CSR *PreT = pre->PreT;

  if(forward){
    CSR *h = Pre;
    Pre = PreT;
    PreT = h;
  }

  vector<vtype> *temp = Vector::init<vtype>(Pre->n, true, true);
  vector<vtype> *p = Vector::init<vtype>(Pre->n, true, true);
  vector<vtype> *Ax = Vector::init<vtype>(Pre->n, true, true);

  CHECK_CUSPARSE(  cusparseDcsrmv(h->cusparse_h0, CUSPARSE_OPERATION_NON_TRANSPOSE, Pre->n, Pre->m, Pre->nnz , &omega, *Pre->descr, Pre->val, Pre->row, Pre->col, rhs->val, &b, temp->val)  );
  CHECK_CUSPARSE(  cusparseDcsrmv(h->cusparse_h0, CUSPARSE_OPERATION_NON_TRANSPOSE, PreT->n, PreT->m, PreT->nnz, &a, *PreT->descr, PreT->val, PreT->row, PreT->col, temp->val, &b, p->val)  );
  CHECK_CUSPARSE(  cusparseDcsrmv(h->cusparse_h0, CUSPARSE_OPERATION_NON_TRANSPOSE, A->n, A->m, A->nnz, &a, *A->descr, A->val, A->row, A->col, p->val, &b, Ax->val)  );

  double ptap, alpha;
  cublasDdot(h->cublas_h, A->n, p->val, 1, Ax->val, 1, &ptap);
  cublasDdot(h->cublas_h, A->n, rhs->val, 1, p->val, 1, &alpha);
  alpha = alpha / ptap;
  cublasDaxpy(h->cublas_h, A->n, &alpha, p->val, 1, x_tent->val, 1);

  Vector::free(temp);
  Vector::free(p);
  Vector::free(Ax);

  return x_tent;
}
*/

vector<vtype>* relax(handles *h, int k, int level, hierarchy *hrrch, vector<vtype> *f, int relax_type, vtype relax_weight, vector<vtype> *u, vector<vtype> **u_, bool forward=true){

  if(relax_type == 0)
    return jacobi_adaptive_miniwarp(h->cusparse_h0, h->cublas_h, k, hrrch->A_array[level], u, u_, f, hrrch->D_array[level], relax_weight);
  else if(relax_type == 1){
    assert(false);
  }else if(relax_type == 2){
    assert(false);
  }else if(relax_type == 4){
    return jacobi_adaptive_miniwarp(h->cusparse_h0, h->cublas_h, k, hrrch->A_array[level], u, u_, f, hrrch->M_array[level], relax_weight);
  }else if(relax_type == 5){
#if AFSAI == 1
    return aFSAI_smoother(h, hrrch->A_array[level], hrrch->pre_array[level], f, u, *u_, forward, k);
    //return naive_aFSAI_smoother(h, hrrch->A_array[level], hrrch->pre_array[level], f, u, forward, k);
    //Vector::print(u);
  //return jacobi_adaptive_miniwarp(h->cusparse_h0, h->cublas_h, k, hrrch->A_array[level], u, u_, f, hrrch->M_array[level], relax_weight);
#endif
}

  return NULL;
}
