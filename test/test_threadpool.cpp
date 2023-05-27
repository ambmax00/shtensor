#include "ThreadPool.h"
#include "TestUtils.h"

#include <chrono>
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
    Shtensor::ThreadPool tpool(4);  

    std::atomic<int> counter = 0;

    const int64_t start = 5;
    const int64_t end = 263;
    const int64_t step = 2;

    tpool.run(start, end, step, 
      [&counter,step]([[maybe_unused]]int64_t _id)
      { 
        // make first 50 steps take longer
        if (_id < 50) 
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        counter += step; 
      });

    SHTENSOR_TEST_EQUAL(counter.load(), end-start, result);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return result;
}