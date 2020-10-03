#if KERNEL_TYPE == 0
  #define makeC_BLOCKSIZE 1024
  #define auction_BLOCKSIZE 1024
  #define assignWtoM_BLOCKSIZE 1024
  #define makeAH_BLOCKSIZE 1024
  #define make_c_BLOCKSIZE 1024
  #define make_w_BLOCKSIZE 1024
  #define write_T_BLOCKSIZE 1024
#elif KERNEL_TYPE == 1
  #define write_T_BLOCKSIZE 1024
  #define makeC_BLOCKSIZE 1024
  #define auction_BLOCKSIZE 1024
  #define assignWtoM_BLOCKSIZE 1024
  #define makeAH_BLOCKSIZE 1024
  #define make_c_BLOCKSIZE 1024
  #define make_w_BLOCKSIZE 1024
  #define aggregate_unsymmetric_BLOCKSIZE 1024
#endif

// Matrix's value type
#define vtype double
// Matrix's index type
#define itype  int
// Matrix's sizes  type
#define stype  int
