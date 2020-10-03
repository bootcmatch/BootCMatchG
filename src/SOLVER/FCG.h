#pragma once

struct FCGPreconditionContext{
  vectorCollection<vtype> *RHS_buffer;
  vectorCollection<vtype> *Xtent_buffer;
  vectorCollection<vtype> *Xtent_buffer_2;
  
  hierarchy *hrrch;
  int max_level_nums;
  itype *max_coarse_size;
};


namespace FCGK{

  FCGPreconditionContext context;

  void initPreconditionContext(hierarchy *hrrch){

    FCGK::context.hrrch = hrrch;
    int num_levels = hrrch->num_levels;

    FCGK::context.max_level_nums = num_levels;
    FCGK::context.max_coarse_size = (itype*) malloc( num_levels * sizeof(int));
    assert(FCGK::context.max_coarse_size != NULL);

    vectorCollection<vtype> *RHS_buffer = Vector::Collection::init<vtype>(num_levels);
    vectorCollection<vtype> *Xtent_buffer = Vector::Collection::init<vtype>(num_levels);

    // !skip the first
    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;
      FCGK::context.max_coarse_size[i] = n_i;
      RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
      Xtent_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
    }

    FCGK::context.RHS_buffer = RHS_buffer;
    FCGK::context.Xtent_buffer = Xtent_buffer;
  }

  void setHrrchBufferSize(hierarchy *hrrch){
    int num_levels = hrrch->num_levels;
    assert(num_levels <= FCGK::context.max_level_nums);

    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;

      if(n_i > FCGK::context.max_coarse_size[i]){
        // make i-level's buffer bigger

        FCGK::context.max_coarse_size[i] = n_i;
        Vector::free(FCGK::context.RHS_buffer->val[i]);
        FCGK::context.RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
        Vector::free(FCGK::context.Xtent_buffer->val[i]);
        FCGK::context.Xtent_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
      }else{
        FCGK::context.RHS_buffer->val[i]->n = n_i;
        FCGK::context.Xtent_buffer->val[i]->n = n_i;
      }
    }
  }

  void freePreconditionContext(){
    free(FCGK::context.max_coarse_size);
    Vector::Collection::free(FCGK::context.RHS_buffer);
    Vector::Collection::free(FCGK::context.Xtent_buffer);
  }
}
//###########################################################################

namespace FCG{

  FCGPreconditionContext context;

  void initPreconditionContext(hierarchy *hrrch){

    FCG::context.hrrch = hrrch;
    int num_levels = hrrch->num_levels;

    FCG::context.max_level_nums = num_levels;
    FCG::context.max_coarse_size = (itype*) malloc( num_levels * sizeof(int));
    assert(FCG::context.max_coarse_size != NULL);

    vectorCollection<vtype> *RHS_buffer = Vector::Collection::init<vtype>(num_levels);
    vectorCollection<vtype> *Xtent_buffer = Vector::Collection::init<vtype>(num_levels);
    vectorCollection<vtype> *Xtent_buffer_2 = Vector::Collection::init<vtype>(num_levels);

    // !skip the first
    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;
      FCG::context.max_coarse_size[i] = n_i;
      RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
      Xtent_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
      Xtent_buffer_2->val[i] = Vector::init<vtype>(n_i, true, true);
    }

    FCG::context.RHS_buffer = RHS_buffer;
    FCG::context.Xtent_buffer = Xtent_buffer;
    FCG::context.Xtent_buffer_2 = Xtent_buffer_2;
  }

  void setHrrchBufferSize(hierarchy *hrrch){
    int num_levels = hrrch->num_levels;
    assert(num_levels <= FCG::context.max_level_nums);

    for(int i=0; i<num_levels; i++){
      itype n_i = hrrch->A_array[i]->n;

      if(n_i > FCG::context.max_coarse_size[i]){
        // make i-level's buffer bigger

        FCG::context.max_coarse_size[i] = n_i;
        Vector::free(FCG::context.RHS_buffer->val[i]);
        FCG::context.RHS_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
        Vector::free(FCG::context.Xtent_buffer->val[i]);
        FCG::context.Xtent_buffer->val[i] = Vector::init<vtype>(n_i, true, true);
        Vector::free(FCG::context.Xtent_buffer_2->val[i]);
        FCG::context.Xtent_buffer_2->val[i] = Vector::init<vtype>(n_i, true, true);
      }else{
        FCG::context.RHS_buffer->val[i]->n = n_i;
        FCG::context.Xtent_buffer->val[i]->n = n_i;
        FCG::context.Xtent_buffer_2->val[i]->n = n_i;
      }
    }
  }

  void freePreconditionContext(){
    free(FCG::context.max_coarse_size);
    Vector::Collection::free(FCG::context.RHS_buffer);
    Vector::Collection::free(FCG::context.Xtent_buffer);
    Vector::Collection::free(FCG::context.Xtent_buffer_2);
  }
}

//#####################################################################################

void inneritkcycle(handles *h, int kh, vector<vtype> *x, vector<vtype> *rhs, bootBuildData  *bootamg_data, boot * boot_amg, applyData *amg_cycle, double rtol, int l);
