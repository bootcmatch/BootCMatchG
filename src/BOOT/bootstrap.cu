#pragma once

namespace Bootstrap{

    void innerIterations(handles *h, bootBuildData *bootamg_data, boot *boot_amg, applyData *amg_cycle){

      buildData *amg_data = bootamg_data->amg_data;
      CSR *A = amg_data->A;
      vector<vtype> *w = amg_data->w;
      // current solution: at the end it will point to the new smooth vector */
      vector<vtype> *x = Vector::init<vtype>(A->n, true, true);
      //NB: Vector::fillWithRandomValue(h->uniformRNG, x);
      Vector::fillWithValue(x, 1.);

      // rhs
      vector<vtype> *rhs = Vector::init<vtype>(A->n, true, true);
      Vector::fillWithValue(rhs, 0.);

      vtype normold = CSRm::vectorANorm(h->cusparse_h0, h->cublas_h, A, x);
      vtype normnew;

      vtype conv_ratio;

      for(int i=1; i<=bootamg_data->solver_it; i++){
        preconditionApply(h, bootamg_data, boot_amg, amg_cycle, rhs, x);
        normnew = CSRm::vectorANorm(h->cusparse_h0, h->cublas_h, A, x);
        conv_ratio = normnew / normold;
        normold = normnew;
      }

      std::cout << "\n conv_ratio " << conv_ratio << "\n";

      vtype alpha = 1. / normnew;

      printf("current smooth vector A-norm=%e\n", normnew);

      Vector::scale(h->cublas_h, x, alpha);
      Vector::copyTo(w, x);

      boot_amg->estimated_ratio = conv_ratio;

      Vector::free(x);
      Vector::free(rhs);
  }

  boot* bootstrap(handles *h, bootBuildData *bootamg_data, applyData *apply_data){

    boot *boot_amg = AMG::Boot::init(bootamg_data->max_hrc, 1.0);

    buildData *amg_data;
    amg_data = bootamg_data->amg_data;

    int num_hrc = 0;

    //for(num_hrc=0; num_hrc<bootamg_data->max_hrc; num_hrc++){
    while(boot_amg->estimated_ratio > bootamg_data->conv_ratio && num_hrc < bootamg_data->max_hrc){

      boot_amg->H_array[num_hrc] = adaptiveCoarsening(h, amg_data); /* this is always done (look at AMG folder) */
      num_hrc++;
      boot_amg->n_hrc = num_hrc;

      if(VERBOSE > 0)
        printf("Built new hierarchy. Current number of hierarchies:%d\n", num_hrc);

      //vector<vtype> *w = amg_data->w;
      //Vector::print(w, 100);

      if(num_hrc == 1){
        // init FGC buffers
        FCG::initPreconditionContext(boot_amg->H_array[0]); /* this is always done */
        if(apply_data->cycle_type == 3) 
          FCGK::initPreconditionContext(boot_amg->H_array[0]);
      }else{
        innerIterations(h, bootamg_data, boot_amg, apply_data);
      }

      //Vector::print(w, 100);
      std::cout << "estimated_ratio " << boot_amg->estimated_ratio << "\n";
    }

    AMG::Boot::finalize(boot_amg, num_hrc);

    std::cout << "Number of hierarchy: " << num_hrc << "\n";
    for(int j=0; j<num_hrc; j++){
      hierarchy *h = boot_amg->H_array[j];
      printf("%d] %d\n", j, h->A_array[h->num_levels-1]->n);
    }

    return boot_amg;
  }

}
