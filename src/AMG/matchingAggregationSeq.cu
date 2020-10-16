
CSR* matchingPairAggregationSeq(CSR *A_, vector<vtype> *w_, vector<itype> *M_, bool symmetric){

  CSR *A = CSRm::copyToHost(A_);
   vector<vtype> *w = Vector::copyToHost(w_);
   vector<itype> *M = Vector::copyToHost(M_);

  int nrows_A = A->n;
  int i, j;
  double wagg0, wagg1, normwagg;
  int ncolc = 0, npairs = 0, nsingle = 0;

  int *p = M->val;
  double *w_data = w->val;

  int *markc = (int *) calloc(nrows_A, sizeof(int));
  for(i=0; i<nrows_A; ++i) markc[i]=-1;


  double *wtempc = (double *) calloc(nrows_A, sizeof(double));

  for(i=0; i<nrows_A; ++i)
    {
      j=p[i];

      if((j>=0) && (i != j))
	{
	  if(markc[i] == -1 && markc[j]== -1)
	    {
	      wagg0=w_data[i];
	      wagg1=w_data[j];
	      normwagg=sqrt(pow(wagg0,2)+pow(wagg1,2));
//	      using the De-norm
	      //normwagg=sqrt(D_data[i]*pow(wagg0,2)+D_data[j]*pow(wagg1,2));
	      if(normwagg > DBL_EPSILON)
		{
		  markc[i]=ncolc;
		  markc[j]=ncolc;
		  wtempc[i]=w_data[i]/normwagg;
		  wtempc[j]=w_data[j]/normwagg;
		  ncolc++;
		  npairs++;
		}
	    }
	}
    }

  for(i=0; i<nrows_A; ++i)
    {
      if(markc[i]==-1)
	{
	  if(fabs(w_data[i]) <= DBL_EPSILON)
	    {
	      /* only-fine grid node: corresponding null row in the prolongator */
	      markc[i]=ncolc-1;
	      wtempc[i]=0.0;
	    }
	  else
	    {
	      markc[i]=ncolc;
	      ncolc++;
	      wtempc[i]=w_data[i]/fabs(w_data[i]);
	      nsingle++;
	    }
	}
    }


  int ncoarse=npairs+nsingle;

  assert(ncolc == ncoarse);

  /* Each row of P has only 1 element. It can be zero in case
     of only-fine grid variable or singleton */

  CSR *P = CSRm::init(nrows_A, ncolc, nrows_A, true, false, false);

  if (ncoarse > 0){
    for(i=0; i<nrows_A; i++){
  	  P->row[i]=i;
  	  P->col[i]=markc[i];
  	  P->val[i]=wtempc[i];
	   }
    P->row[nrows_A]=nrows_A;
  }

  //free(markc);
  //free(wtempc);

  //CSRm::print(P, 1);
  //CSRm::print(P, 0);

  return P;
}
