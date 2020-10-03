#include "matrix/CSR.h"

#define INVASIVE_DEBUG 0

#pragma once

void SETUP_KERNEL_CACHE(){
  cudaFuncSetCacheConfig(CSRm::_CSR_vector_mul_mini_warp<0>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(CSRm::_CSR_vector_mul_mini_warp<1>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(CSRm::_CSR_vector_mul_mini_warp<2>, cudaFuncCachePreferL1);

  cudaFuncSetCacheConfig(_jacobi_it<0, 2>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_jacobi_it<0, 4>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_jacobi_it<0, 8>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_jacobi_it<0, 16>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(_jacobi_it<0, 32>, cudaFuncCachePreferL1);
}

// Test/testkbootsolvers.c
vector<vtype>* solve(CSR *A_host, const params p, vector<vtype> *rhs){

  SETUP_KERNEL_CACHE();

  //AMG::Params::metaPrintInfo(p);

  handles *h = Handles::init();

  TIME::start();
  CSR *A = CSRm::copyToDevice(A_host);

//  CSRm::powerMethod(h->cusparse_h0, h->cublas_h, A, 0.001, 10000);

  float load_time = TIME::stop();

  std::cout << "BUILDING....:\n\n";
  bootBuildData *bootamg_data;
  bootamg_data = AMG::BootBuildData::initByParams(A, p);

  buildData *amg_data;
  amg_data = bootamg_data->amg_data;

  applyData *amg_cycle;
  amg_cycle = AMG::ApplyData::initByParams(p);
  AMG::ApplyData::setGridSweeps(amg_cycle, amg_data->maxlevels);

  if(VERBOSE > 0){
    AMG::BootBuildData::print(bootamg_data);
    AMG::BuildData::print(amg_data);
    AMG::ApplyData::print(amg_cycle);
  }

  Relax::initContext(A->n);
  GAMGcycle::initContext(A->n);

  // start bootstrap process
  TIME::start();

#if MANUAL_PROF_TYPE == 1
  cudaProfilerStart();
#endif

  boot *boot_amg = Bootstrap::bootstrap(h, bootamg_data, amg_cycle);
  float boot_time = TIME::stop();

#if MANUAL_PROF_TYPE == 1
  cudaProfilerStop();
#endif

  //-------------------------------------------------------------------

#if INVASIVE_DEBUG
  CSRmInfo::printMeta(boot_amg->H_array[0]->A_array, boot_amg->H_array[0]->num_levels, "A");
  CSRmInfo::printMeta(boot_amg->H_array[0]->P_array, boot_amg->H_array[0]->num_levels - 1, "P");
  CSRmInfo::printMeta(boot_amg->H_array[0]->R_array, boot_amg->H_array[0]->num_levels - 1, "R");
#endif

  int precon = 1;
  int num_iter = 0;

  vector<vtype> *Sol = Vector::init<vtype>(A->n, true, true);
  Vector::fillWithValue(Sol, 0.);

#if MANUAL_PROF_TYPE == 2
  cudaProfilerStart();
#endif


  std::cout << "SOLVING....:\n\n";

  TIME::start();
  vtype residual = 0.;
#if CG_VERSION == 0
  residual = flexibileConjugateGradients(h, Sol, rhs, bootamg_data, boot_amg, amg_cycle, precon, p.itnlim, p.rtol, &num_iter);
#elif CG_VERSION == 1
  residual = flexibileConjugateGradients_v2(h, Sol, rhs, bootamg_data, boot_amg, amg_cycle, precon, p.itnlim, p.rtol, &num_iter);
#elif CG_VERSION == 2
  residual = flexibileConjugateGradients_v3(h, Sol, rhs, bootamg_data, boot_amg, amg_cycle, precon, p.itnlim, p.rtol, &num_iter);
#endif

  float solve_time = TIME::stop();

#if MANUAL_PROF_TYPE == 2
  cudaProfilerStop();
#endif

  Eval::printMetaData("agg;hierarchy_levels_num", boot_amg->n_hrc, 0);
  Eval::printMetaData("agg;final_estimated_ratio", boot_amg->estimated_ratio, 1);
  Eval::printMetaData("sol;num_iteration", num_iter, 0);
  Eval::printMetaData("sol;residual", residual, 1);
  //Eval::printMetaData("time;matrix_device_copy_time", load_time, 1);
  Eval::printMetaData("time;bootstrap_time", boot_time, 1);
  Eval::printMetaData("time;solve_time", solve_time, 1);

  std::cout << "bootstrap_time: " << (boot_time / 1000) << " sec \n";
  std::cout << "solve_time: " << (solve_time / 1000) << " sec \n";

  float time_for_iteration = solve_time / num_iter;
  std::cout << "time_for_iteration: " << (time_for_iteration / 1000) << " sec \n";

  CSRm::free(A);
  AMG::Boot::free(boot_amg);
  AMG::BootBuildData::free(bootamg_data);
  AMG::ApplyData::free(amg_cycle);
  Relax::freeContext();
  GAMGcycle::freeContext();
  Handles::free(h);

  return Sol;
}
