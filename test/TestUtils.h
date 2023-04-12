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
    printf("Error: " #a " and " #b " not equal\n");\
    result+=1;\
  }

#define SHTENSOR_TEST_TRUE(expr,result) \
  if (!expr)\
  {\
    result+=1;\
  }

//#define SHTENSOR_TEST_ALMOST_EQUAL