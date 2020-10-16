#define AFSAI 0

#define MATCHING_MOD_TYPE 0
// 1 FORCE A TO BE SYM 0 FORCE M TO BE SYM
#define TYPE_WRITE_T 1

#define MATRIX_MATRIX_MUL_TYPE 1
// 0 cuSPARSE 1 nsparse

#define CG_VERSION 2

#define CSR_JACOBI_TYPE 0
// 0 Se applicare JACOBY hybrid con cuSPARSE o  1 JACOBI AHMW

#define CSR_VECTOR_MUL_GENERAL_TYPE 1
#define CSR_VECTOR_MUL_A_TYPE 1
#define CSR_VECTOR_MUL_P_TYPE 1
#define CSR_VECTOR_MUL_R_TYPE 1

#define MAXIMUM_PRODUCT_MATRIX_OP 1
#define GALERKIN_PRODUCT_TYPE 1


#include <iostream>
#include <cuda.h>
#include <cmath>
#include <cfloat>

#include "matrix/CSR.h"
#include "matrix/scalar.cu"

#include "matrix/matrixIO.h"

#include "utility/eval.cu"
#include "AMG/utility.cu"
#include "utility/handles.cu"
#include "AMG/AMG.cu"

#include "utility/setting.h"
#include "utility/eval.cu"

#include "AMG/matching.cu"
#include "AMG/matchingAggregation.cu"
#include "AMG/matchingAggregationSeq.cu"

#include "SOLVER/GAMG_cycle.cu"
#include "SOLVER/FCG.cu"
#include "BOOT/bootstrap.cu"
#include "SOLVER/solver.cu"
