#define aggregateunsym_BLOCKSIZE 1024

__global__
void _setUnsymPair(int n, int *M, int *Msym){
  int v = blockDim.x * blockIdx.x + threadIdx.x;

  if(v >= n)
    return;

  int u = M[v];

  if(u == -1 || Msym[v] != -1 || u > v)
    return;

  rep:

  if( atomicCAS(&Msym[u], -1, v) == -1 ){
    int u_temp = atomicCAS(&Msym[v], -1, u);
    if(u_temp != -1){
      // re-match
      Msym[u] = -1;
      if(v > u) {
           return;
      }
      goto rep;
    }
    // win
  }
  // lose
}

__global__
void _setSymPair(itype n, itype *M, itype *Msym){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= n)
    return;

  itype v = tid;
  itype u = M[v];

  if(u == -1)
    return;

  // sym pair
  if(v == M[u]){
    Msym[v] = u;
    Msym[u] = v;
  }
}



//###############################MAIN###########################################
//##############################################################################
//##############################################################################

vector<itype>* unsymFix(vector<itype> *M){

  itype n = M->n;

  vector<int> *Msym = Vector::init<int>(n, true, true);
  Vector::fillWithValue(Msym, -1);

  gridblock gb = gb1d(n, aggregateunsym_BLOCKSIZE);

  _setSymPair<<<gb.g, gb.b>>>(n, M->val, Msym->val);
  _setUnsymPair<<<gb.g, gb.b>>>(n, M->val, Msym->val);

  return Msym;
}
