#pragma once

#include "../SOLVER/LU.cu"
#if AFSAI == 1
#include "myAFSAI/afsai.h"
#endif


struct params{
  const char *rhsfile;
  const char *solfile;
  int solver_type;
  int max_hrc;
  double conv_ratio;
  int matchtype;
  int aggrsweeps;
  int aggrtype;
  int max_levels;
  int cycle_type;
  int coarse_solver;
  int relax_type;
  int prerelax_sweeps;
  int postrelax_sweeps;
  int itnlim;
  double rtol;

  int relaxnumber_coarse;
};

struct hierarchy{
  CSR **A_array; /*array of coarse matrices including the fine ones*/
  CSR **P_array; /*array of prolongators */
  CSR **R_array; /*array of prolongators */
  vector<vtype> **D_array; /* diagonal of coarse matrices */
  vector<vtype> **M_array; /* new diagonal matrix */

#if AFSAI == 1
  PRECOND **pre_array;
#endif

  LU_factor *LU;

  itype num_levels; /* number of levels of the hierachy */
  vtype op_cmplx;  /* operator complexity of the hierarchy for V-cycle*/
  vtype op_wcmplx;  /* operator complexity of the hierarchy for W-cycle*/
  vtype avg_cratio;  /* average of coarsening ratio of the hierarchy */
};

struct buildData{
  /* setup params */
  itype maxlevels; /* max number of levels per hierarchy */
  itype maxcoarsesize; /* maximum size of the coarsest matrix */
  itype sweepnumber; /* number of pairwise aggregation steps. Currently 0 for pairwise and 1 for double pairwise */
  itype agg_interp_type; /* 1 for smoothed aggregation, 0 for pure aggregation */
  itype coarse_solver; /* solver to be used on the coarsest level */

  /* CR params */
  itype CRrelax_type; /* to choose relaxation scheme for Compatible Relaxation */
  vtype CRrelax_weight; /* weight for weighted Jacobi in CR */
  itype CRit;  /* number of iterations for Compatible Relaxation */
  vtype CRratio; /* optimal convergence ratio in Compatible Relaxation to stop coarsening*/

  CSR *A;
  vector<vtype> *w;

  itype ftcoarse;
};

struct applyData{
  /* cycle type params */
  int cycle_type; /* 0 for V-cycle, 2 for W-cycle, 1 for G-cycle */
  int *num_grid_sweeps; /* number of sweeps on a fixed level in case of G-cycle */

  int relax_type; /* type of pre/post relaxation/smoothing */
  int relaxnumber_coarse;
  int prerelax_number; /* number of pre smoothing steps */
  int postrelax_number; /* number of post smoothing steps */

  vtype relax_weight; /* weight for Jacobi relaxation */
};


struct bootBuildData{
   /* setup params for bootstrap process */
   int max_hrc; /* max number of hierarchies to be built*/
   double conv_ratio; /* desired convergence ratio */
   int solver_type; /* type of composition for applying composite solver */
  /* solver_type =0 -> multiplicative composition
   * solver_type =1 -> symmetrized multiplicative composition
   * solver_type =2 -> additive composition */
   int solver_it; /* number of iterations to be applied for conv. ratio estimating */

  /* setup params per each AMG component */
   buildData *amg_data;
};

struct boot{
   /* data generated in the setup phase */
   hierarchy **H_array; /*array of AMG hierarchies */
   int n_hrc; /* number of hierarchies  */
   double estimated_ratio; /* estimated convergence ratio obtained after built*/
};

namespace AMG{

  namespace Hierarchy{

    hierarchy* init(itype num_levels, bool allocate_mem=true){

      hierarchy *H = NULL;
      // on the host
      H = (hierarchy*) malloc(sizeof(hierarchy));
      CHECK_HOST(H);

      H->num_levels = num_levels;
      H->op_cmplx = 0;

      H->A_array = NULL;
      H->P_array = NULL;
      H->R_array = NULL;
  #if AFSAI == 1
      H->pre_array = NULL;
  #endif

      if(allocate_mem){
        H->A_array = (CSR**) malloc(num_levels * sizeof(CSR*));
        CHECK_HOST(H->A_array);

        H->P_array = (CSR**) malloc( (num_levels-1) * sizeof(CSR*) );
        CHECK_HOST(H->P_array);

        H->R_array = (CSR**) malloc( (num_levels-1) * sizeof(CSR*) );
        CHECK_HOST(H->R_array);

        H->D_array = (vector<vtype>**) malloc( num_levels * sizeof(vector<vtype>*));
        CHECK_HOST(H->D_array);

        H->M_array = (vector<vtype>**) malloc( num_levels * sizeof(vector<vtype>*));
        CHECK_HOST(H->M_array);

#if AFSAI == 1
        H->pre_array = (PRECOND**) malloc( num_levels * sizeof(PRECOND*) );
        CHECK_HOST(H->pre_array);
#endif

        for(int i=0; i<H->num_levels; i++){
          H->D_array[i] = NULL;
          if(i != H->num_levels - 1)
            H->R_array[i] = NULL;
          H->M_array[i] = NULL;
#if AFSAI == 1
          H->pre_array[i] = NULL;
#endif
        }
      }

      return H;
    }

    void free(hierarchy *H){

      for(int i=0; i<H->num_levels; i++){

        // skip the original matrix
        if(i > 0)
          CSRm::free(H->A_array[i]);

        if( H->D_array[i] != NULL )
          Vector::free(H->D_array[i]);

        if( H->M_array[i] != NULL )
          Vector::free(H->M_array[i]);


        if(i != H->num_levels - 1){
            CSRm::free(H->P_array[i]);
            if( H->R_array[i] != NULL )
              CSRm::free(H->R_array[i]);
        }
      }

      std::free(H->A_array);
      std::free(H->D_array);
      std::free(H->M_array);
      std::free(H->P_array);

      std::free(H);
    }

    void finalize_level(hierarchy *H, int levels_used){

      assert(levels_used > 0);

      H->num_levels = levels_used;

      H->A_array = (CSR**) realloc(H->A_array, levels_used * sizeof(CSR*));
      CHECK_HOST(H->A_array);

      H->P_array = (CSR**) realloc(H->P_array, (levels_used-1) * sizeof(CSR*) );
      CHECK_HOST(H->P_array);

      H->R_array = (CSR**) realloc(H->R_array, (levels_used-1) * sizeof(CSR*) );
      CHECK_HOST(H->R_array);

      H->D_array = (vector<vtype>**) realloc(H->D_array, levels_used * sizeof(vector<vtype>*));
      CHECK_HOST(H->D_array);

      H->M_array = (vector<vtype>**) realloc(H->M_array, levels_used * sizeof(vector<vtype>*));
      CHECK_HOST(H->M_array);

    }

    vtype finalize_cmplx(hierarchy *h){
      vtype cmplxfinal = 0;
      for(int i=0; i<h->num_levels; i++){
        cmplxfinal += h->A_array[i]->nnz;
      }
      cmplxfinal /= h->A_array[0]->nnz;
      h->op_cmplx = cmplxfinal;
      return cmplxfinal;
    }

    vtype finalize_wcmplx(hierarchy *h){
      vtype wcmplxfinal = 0;
      for(int i=0; i<h->num_levels; i++){
        wcmplxfinal +=  pow(2, i) * h->A_array[i]->nnz;
      }
      wcmplxfinal /= h->A_array[0]->nnz;
      h->op_wcmplx = wcmplxfinal;
      return wcmplxfinal;
    }

    void printInfo(hierarchy *h){

      for(int i=0; i<h->num_levels; i++){
        CSR *Ai = h->A_array[i];
        float avg_nnz = (float) Ai->nnz / (float) Ai->n;
        std::cout << "A" << i << " n: " << Ai->n << " nnz: " << Ai->nnz << " avg_nnz: " << avg_nnz << "\n";
      }
      std::cout << "\nCurrent cmplx for V-cycle: " << h->op_cmplx;
      std::cout << "\nCurrent cmplx for W-cycle: " << h->op_wcmplx;
      std::cout << "\nAverage Coarsening Ratio: " << h->avg_cratio << "\n";
    }
  }

  namespace BuildData{
    buildData* init(itype maxlevels, itype maxcoarsesize, itype sweepnumber, itype agg_interp_type, itype coarse_solver, itype CRrelax_type, vtype CRrelax_weight, itype CRit, vtype CRratio){
      buildData *bd = NULL;

      bd = (buildData*) malloc(sizeof(buildData));
      CHECK_HOST(bd);

      bd->maxlevels = maxlevels;
      bd->maxcoarsesize = maxcoarsesize;
      bd->sweepnumber = sweepnumber;
      bd->agg_interp_type = agg_interp_type;
      bd->coarse_solver = coarse_solver;
      bd->CRrelax_type = CRrelax_type;
      bd->CRrelax_weight = CRrelax_weight;
      bd->CRit = CRit;
      bd->CRratio = CRratio;

      bd->A = NULL;
      bd->w = NULL;

      bd->ftcoarse = 1;

      return bd;
    }

    void free(buildData *bd){
      //Vector::free(bd->w);
      std::free(bd);
    }

    buildData* initDefault(){
      buildData *bd = NULL;

      bd = (buildData*) malloc(sizeof(buildData));
      CHECK_HOST(bd);

      bd->maxlevels = 100;
      bd->maxcoarsesize = 100;
      bd->sweepnumber = 1;
      bd->agg_interp_type = 0;
      bd->coarse_solver = 9;

      bd->CRrelax_type = 0;
      bd->CRrelax_weight = 1. / 3.;
      bd->CRit = 0;
      bd->CRratio = .3;

      bd->A = NULL;
      bd->w = NULL;

      bd->ftcoarse = 1;

      return bd;
    }

    void setMaxCoarseSize(buildData *bd){
      bd->maxcoarsesize = (itype) ( 40 * pow( (double) bd->A->n, (double)1. / 3.) );
    }

    void print(buildData *bd){
      std::cout << "\nmaxlevels: " << bd->maxlevels << "\n";
      std::cout << "maxcoarsesize: " << bd->maxcoarsesize << "\n";
      std::cout << "sweepnumber: " << bd->sweepnumber << "\n";
      std::cout << "agg_interp_type: " << bd->agg_interp_type << "\n";
      std::cout << "coarse_solver: " << bd->coarse_solver << "\n";
      std::cout << "CRrelax_type: " << bd->CRrelax_type << "\n";
      std::cout << "CRrelax_weight: " << bd->CRrelax_weight << "\n";
      std::cout << "CRit: " << bd->CRit << "\n";
      std::cout << "CRratio: " << bd->CRratio << "\n";
      std::cout << "ftcoarse: " << bd->ftcoarse << "\n\n";
    }
  }

  namespace ApplyData{
    applyData* initDefault(){

      applyData *ad = NULL;
      ad = (applyData*) malloc(sizeof(applyData));
      CHECK_HOST(ad);

      ad->cycle_type = 0;
      ad->relax_type = 0;
      ad->relaxnumber_coarse = 1;
      ad->prerelax_number = 1;
      ad->postrelax_number = 1;
      ad->relax_weight = 1.0;
      ad->num_grid_sweeps = NULL;

      return ad;
    }

    void free(applyData *ad){
      std::free(ad->num_grid_sweeps);
      std::free(ad);
    }

    void print(applyData *ad){
      std::cout << "\ncycle_type: " << ad->cycle_type << "\n";
      std::cout << "relax_type: " << ad->relax_type << "\n";
      std::cout << "relaxnumber_coarse: " << ad->relaxnumber_coarse << "\n";
      std::cout << "prerelax_number: " << ad->prerelax_number << "\n";
      std::cout << "postrelax_number: " << ad->postrelax_number << "\n";
      std::cout << "relax_weight: " << ad->relax_weight << "\n\n";
    }

    applyData* initByParams(params p){
      applyData *amg_cycle = AMG::ApplyData::initDefault();

      amg_cycle->cycle_type = p.cycle_type;
      amg_cycle->relax_type = p.relax_type;
      amg_cycle->relaxnumber_coarse = p.relaxnumber_coarse;
      amg_cycle->prerelax_number = p.prerelax_sweeps;
      amg_cycle->postrelax_number = p.postrelax_sweeps;

      return amg_cycle;
    }

    void setGridSweeps(applyData *ad, int max_level){
      max_level--;
      ad->num_grid_sweeps = (int*) malloc( max_level * sizeof(int));
      CHECK_HOST(ad->num_grid_sweeps);

      int i, j;

      for(i=0; i<max_level; i++)
        ad->num_grid_sweeps[i] = 1;

      if(ad->cycle_type == 1){
        // H-cycle
        for(i=0; i<max_level; i++){
          j = i % 2; /*step is fixed to 2; it can be also different */
          if(j == 0)
            ad->num_grid_sweeps[i] = 2;
        }
      }else if(ad->cycle_type == 2){
        // W-cycle
        for(i=0; i<max_level-1; i++){
          ad->num_grid_sweeps[i] = 2;
        }
      }
    }
  }

  namespace BootBuildData{
    bootBuildData* initDefault(){

      bootBuildData *bd = NULL;
      bd = (bootBuildData*) malloc(sizeof(bootBuildData));
      CHECK_HOST(bd);

      bd->max_hrc = 10;
      bd->conv_ratio = 0.80;
      bd->solver_type = 1;
      bd->solver_it = 15;

      bd->amg_data = AMG::BuildData::initDefault();

      return bd;
    }

    void free(bootBuildData *ad){
      AMG::BuildData::free(ad->amg_data);
      std::free(ad);
    }

    void print(bootBuildData *ad){
      std::cout << "\nmax_hrc: " << ad->max_hrc << "\n";
      std::cout << "conv_ratio: " << ad->conv_ratio << "\n";
      std::cout << "solver_type: " << ad->solver_type << "\n";
      std::cout << "solver_it: " << ad->solver_it << "\n\n";
    }

    bootBuildData* initByParams(CSR *A, params p){
      bootBuildData *bootamg_data = AMG::BootBuildData::initDefault();
      buildData *amg_data = bootamg_data->amg_data;

      bootamg_data->solver_type = p.solver_type;
      bootamg_data->max_hrc = p.max_hrc;
      bootamg_data->conv_ratio = p.conv_ratio;

      amg_data->sweepnumber = p.aggrsweeps;
      amg_data->agg_interp_type = p.aggrtype;
      amg_data->maxlevels = p.max_levels;
      amg_data->coarse_solver = p.coarse_solver;
      amg_data->CRrelax_type = p.relax_type;


      amg_data->A = A;
      amg_data->w = Vector::init<vtype>(A->n, true, true);
      Vector::fillWithValue(amg_data->w, 1.0);

      AMG::BuildData::setMaxCoarseSize(amg_data);

      return bootamg_data;
    }
  }

  namespace Boot{

    boot* init(int n_hrc, double estimated_ratio){
      boot *b = (boot*) malloc(sizeof(boot));
      CHECK_HOST(b);

      b->n_hrc = n_hrc;
      b->estimated_ratio = estimated_ratio;
      b->H_array = (hierarchy**) malloc( n_hrc * sizeof(hierarchy*) );
      CHECK_HOST(b->H_array);

      return b;
    }

    void free(boot *b){
      for(int i=0; i<b->n_hrc; i++)
        AMG::Hierarchy::free(b->H_array[i]);
      std::free(b);
    }

    void finalize(boot *b, int num_hrc){
      assert(num_hrc > 0);
      b->n_hrc = num_hrc;
      b->H_array = (hierarchy**) realloc(b->H_array, num_hrc * sizeof(hierarchy*));
      CHECK_HOST(b->H_array);
    }
  }

  namespace Params{
    params initFromFile(const char *path){
      params inparms;

      FILE *fp = fopen(path, "r");

      if(fp == NULL){
        std::cout << "Setting file not found!" << "\n";
        exit(-1);
      }

      inparms.rhsfile           = string_get_fp(fp);
      if (strcmp(inparms.rhsfile,"NONE")==0) {
        inparms.rhsfile=NULL;
      }
      inparms.solfile           = string_get_fp(fp);
      if (strcmp(inparms.solfile,"NONE")==0) {
        inparms.solfile=NULL;
      }
      inparms.solver_type       = int_get_fp(fp);
      inparms.max_hrc           = int_get_fp(fp);
      inparms.conv_ratio        = double_get_fp(fp);
      inparms.matchtype         = int_get_fp(fp);
      inparms.aggrsweeps        = int_get_fp(fp) + 1;
      inparms.aggrtype          = int_get_fp(fp);
      inparms.max_levels        = int_get_fp(fp);
      inparms.cycle_type        = int_get_fp(fp);
      inparms.coarse_solver     = int_get_fp(fp);
      inparms.relax_type        = int_get_fp(fp);
      inparms.relaxnumber_coarse= int_get_fp(fp);
      inparms.prerelax_sweeps   = int_get_fp(fp);
      inparms.postrelax_sweeps  = int_get_fp(fp);
      inparms.itnlim            = int_get_fp(fp);
      inparms.rtol              = double_get_fp(fp);

      fclose(fp);
      return inparms;
    }

    void metaPrintInfo(params inparms){
      Eval::printMetaData("params;solver_type", inparms.solver_type, 0);
      Eval::printMetaData("params;max_hrc", inparms.max_hrc, 0);
      Eval::printMetaData("params;conv_ratio", inparms.conv_ratio, 1);
      Eval::printMetaData("params;matchtype", inparms.matchtype, 0);
      Eval::printMetaData("params;aggrsweeps", inparms.aggrsweeps - 1, 0);
      Eval::printMetaData("params;aggrtype", inparms.aggrtype, 0);
      Eval::printMetaData("params;max_levels", inparms.max_levels, 0);
      Eval::printMetaData("params;cycle_type", inparms.cycle_type, 0);
      Eval::printMetaData("params;coarse_solver", inparms.coarse_solver, 0);
      Eval::printMetaData("params;relax_type", inparms.relax_type, 0);
      Eval::printMetaData("params;relaxnumber_coarse", inparms.relaxnumber_coarse, 0);
      Eval::printMetaData("params;prerelax_sweeps", inparms.prerelax_sweeps, 0);
      Eval::printMetaData("params;postrelax_sweeps", inparms.postrelax_sweeps, 0);
      Eval::printMetaData("params;itnlim", inparms.itnlim, 0);
      Eval::printMetaData("params;rtol", inparms.rtol, 1);
    }
  }
}
