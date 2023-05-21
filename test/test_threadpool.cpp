#include "ThreadPool.h"
#include "TestUtils.h"
#include <mpi.h>

int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);
  
  int result = 0;

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
  {
    Shtensor::ThreadPool tpool;  

    std::atomic<int> counter = 0;

    const int64_t start = 5;
    const int64_t end = 263;
    const int64_t step = 2;

    tpool.run([&counter,step]([[maybe_unused]]int64_t _id){ counter += step; }, start, end, step);

    SHTENSOR_TEST_EQUAL(counter.load(), end-start, result);

  }

  MPI_Finalize();

  return result;
}