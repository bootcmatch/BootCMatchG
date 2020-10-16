
#define VERBOSE 0
#define SOLUTION_FILE "x.txt"

#include "BootCMatchG.h"


int main(int argc, char *argv[]){

  const char* matrix_path = NULL, *params_path = NULL;
  int m_type = 0;
  if(argc >= 3){
    matrix_path = argv[1];
    params_path = argv[2];
  }else{
    std::cout << "matrix_path setting_file\n";
    exit(1);
  }

  // read input matrix`
  std::cout << "READING INPUT MATRIX....:\n\n";
  CSR *A_host = readMatrixFromFile(matrix_path, m_type, false);

  std::cout << "SOLVING....:\n\n";
  // CREATE / READ RHS
  vector<vtype> *rhs = Vector::init<vtype>(A_host->n, true, true);
  Vector::fillWithValue(rhs, 1.);

  params p = AMG::Params::initFromFile(params_path);

  vector<vtype> *x = solve(A_host, p, rhs);
  Vector::printOnFile(x, SOLUTION_FILE);

  Vector::free(x);
  CSRm::free(A_host);

  return 0;
}
