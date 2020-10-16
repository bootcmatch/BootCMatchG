#pragma once

#include "FCG.h"

#define RTOL 0.25

namespace GAMGcycle{
  vector<vtype> *Res_buffer;

  void initContext(int n){
    GAMGcycle::Res_buffer = Vector::init<vtype>(n ,true, true);
  }

  __inline__
  void setBufferSize(itype n){
    GAMGcycle::Res_buffer->n = n;
  }

  void freeContext(){
    Vector::free(GAMGcycle::Res_buffer);
  }
}

// bcm_GAMGCycle
void GAMG_cycle(handles *h, int k, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle, vectorCollection<vtype> *Rhs, vectorCollection<vtype> *Xtent, int l){

  hierarchy *hrrc = boot_amg->H_array[k];
  int relax_type = amg_cycle->relax_type;

  buildData *amg_data = bootamg_data->amg_data;
  int coarse_solver = amg_data->coarse_solver;

  if(VERBOSE > 0)
    std::cout << "GAMGCycle: start of level " << l << "\n";

  if(l == hrrc->num_levels){
    if(coarse_solver == 9){
      //Vector::copyTo(Xtent->val[l-1], Rhs->val[l-1]);
      //LU::solve(h->cusparse_h0, hrrc->LU, Xtent->val[l-1], Rhs->val[l-1]);
    }else{
      // solve
      Xtent->val[l-1] = relax(h, amg_cycle->relaxnumber_coarse, l-1, hrrc, Rhs->val[l-1], coarse_solver, amg_cycle->relax_weight, Xtent->val[l-1], &FCG::context.Xtent_buffer_2->val[l-1]);
    }
  }else{
    // presmoothing steps
    Xtent->val[l-1] = relax(h, amg_cycle->prerelax_number, l-1, hrrc, Rhs->val[l-1], relax_type, amg_cycle->relax_weight, Xtent->val[l-1], &FCG::context.Xtent_buffer_2->val[l-1]);

    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Rhs->val[l-1]);
      std::cout << "RHS at level " << l << " " << tnrm << "\n";
      tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
      std::cout << "After first smoother at level " << l << " XTent " << tnrm << "\n";
    }

    // compute residual
    //vector<vtype> *Res = Vector::clone(Rhs->val[l-1]);
    GAMGcycle::setBufferSize(Rhs->val[l-1]->n);
    vector<vtype> *Res = GAMGcycle::Res_buffer;
    Vector::copyTo(Res, Rhs->val[l-1]);


    // Res = -A * Xtent->val + Res
#if CSR_VECTOR_MUL_A_TYPE == 0
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, hrrc->A_array[l-1], Xtent->val[l-1], Res, false, -1., 1.);
#elif CSR_VECTOR_MUL_A_TYPE == 1
    CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, hrrc->A_array[l-1], Xtent->val[l-1], Res, -1., 1.);
#endif

    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Res);
      std::cout << "Residual at level " << l << " " << tnrm << "\n";
    }

    // restrict residual CSR VEC TRANS
#if CSR_VECTOR_MUL_R_TYPE == 0
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, hrrc->R_array[l-1], Res, Rhs->val[l], false, 1., 0.);
#elif CSR_VECTOR_MUL_R_TYPE == 1
    CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, hrrc->R_array[l-1], Res, Rhs->val[l], 1., 0.);
#endif

    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Rhs->val[l]);
      std::cout << "Next RHSl " << l+1 << " " << tnrm << "\n";
    }

    Vector::fillWithValue(Xtent->val[l], 0.);

    if(amg_cycle->cycle_type == 3){

      if(l == hrrc->num_levels-1){
        GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, Rhs, Xtent, l+1);
      }else{
        // NB: da verificare
        //vector<vtype> *x_host = Vector::copyToHost(Rhs->val[l]);
        //printf("\nx %lf\n\n", x_host->val[0]);
        //Vector::free(x_host);
        //Vector::fillWithValue(Xtent->val[l], 0.);
        inneritkcycle(h, k, Xtent->val[l], Rhs->val[l], bootamg_data, boot_amg, amg_cycle, RTOL, l);
      }

    }else{
      // cycle_type == 0, cycle_type == 1, cycle_type == 2

      for(int i=1; i<=amg_cycle->num_grid_sweeps[l-1]; i++){

        GAMG_cycle(h, k, bootamg_data, boot_amg, amg_cycle, Rhs, Xtent, l+1);
        if(l == hrrc->num_levels-1)
          break;
      }
    }

    // prolongate error
#if CSR_VECTOR_MUL_P_TYPE == 0
    CSRm::CSRVector_product_CUSPARSE(h->cusparse_h0, hrrc->P_array[l-1], Xtent->val[l], Xtent->val[l-1], false, 1., 1.);
#elif CSR_VECTOR_MUL_P_TYPE == 1
    if(amg_data->agg_interp_type == 1)
      CSRm::CSRVector_product_adaptive_miniwarp(h->cusparse_h0, hrrc->P_array[l-1], Xtent->val[l], Xtent->val[l-1], 1., 1.);
    else
      CSRm::CSRVector_product_prolungator(hrrc->P_array[l-1], Xtent->val[l], Xtent->val[l-1]);
#endif

    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
      std::cout << "After recursion at level " << l << " XTent " << tnrm << "\n";
    }

    // postsmoothing steps
    Xtent->val[l-1] = relax(h, amg_cycle->postrelax_number, l-1, hrrc, Rhs->val[l-1], relax_type, amg_cycle->relax_weight, Xtent->val[l-1], &FCG::context.Xtent_buffer_2->val[l-1]);

    if(VERBOSE > 1){
      vtype tnrm = Vector::norm(h->cublas_h, Xtent->val[l-1]);
      std::cout << "After second smoother at level " << l << " XTent " << tnrm << "\n";
    }

    //Vector::free(Res);
  }

  if(VERBOSE > 0)
      std::cout << "GAMGCycle: end of level " << l << "\n";
}

//###########################################################################
