#include <algorithm>

#define SHTENSOR_UNPAREN(expr) \
  expr

#define SHTENSOR_DO_BY_RANK(ctx, expression) \
  for (int i = 0; i < ctx.get_size(); ++i) \
  {\
    if (i == ctx.get_rank())\
    {\
      SHTENSOR_UNPAREN expression;\
    }\
    MPI_Barrier(ctx.get_comm());\
  }

#define SHTENSOR_TEST_EQUAL(a,b,result) \
  if (a != b)\
  {\
    if constexpr (std::is_same<decltype(a), int>::value)\
    {\
      printf("Error: " #a " and " #b " not equal. Expected %d, got %d\n", a, b);\
    }\
    else\
    {\
      printf("Error: " #a " and " #b " not equal. Expected %ld, got %ld\n", a, b);\
    }\
    result+=1;\
  }

#define SHTENSOR_TEST_CONTAINER_EQUAL(A,B,result) \
  if (!std::equal(A.begin(), A.end(), B.begin())) \
  {\
    printf("Error: " #A " and " #B " do not have equal elements\n");\
    result += 1;\
  }

#define SHTENSOR_TEST_TRUE(expr,result) \
  if (!expr)\
  {\
    result+=1;\
  }

//#define SHTENSOR_TEST_ALMOST_EQUAL