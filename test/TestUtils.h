#include <algorithm>
#include "Logger.h"

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
    Shtensor::Log::error(logger, "{} and {} are not equal. Expected {}, got {}\n", #a, #b, a, b);\
    result += 1;\
  }

#define SHTENSOR_TEST_CONTAINER_EQUAL(A,B,result) \
  if (!std::equal(A.begin(), A.end(), B.begin())) \
  {\
    Shtensor::Log::error(logger, "{} and {} do not have equal elements\n", #A, #B);\
    result += 1;\
  }

#define SHTENSOR_TEST_ALMOST_EQUAL(a, b, eps, result) \
  if (std::abs(a-b) > eps)\
  {\
    Shtensor::Log::error(logger, "{} and {} differ. Expected {}, got {}\n", #a, #b, a, b);\
    result += 1;\
  }

#define SHTENSOR_TEST_TRUE(expr,result) \
  if (!expr)\
  {\
    result+=1;\
  }

//#define SHTENSOR_TEST_ALMOST_EQUAL