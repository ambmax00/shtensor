#include "TestUtils.h"
#include "Context.h"
#include "PGAS.h"

int main(int argc, char** argv)
{

  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &provided);

  int result = 0;

  {
   
    Shtensor::Context ctx(MPI_COMM_WORLD, 10*Shtensor::MiB);

    auto logger = Shtensor::Log::create(fmt::format("Test-{}", ctx.get_rank()));

    Shtensor::Log::info(logger, "Hello from rank {} on host {}", 
                        ctx.get_rank(),  ctx.get_host_name());

    auto shmem = Shtensor::ShmemInterface{ctx};

    Shtensor::Window<int> win = shmem.allocate<int>(1000);

    const int left = ctx.get_left_neighbour();
    const int right = ctx.get_right_neighbour();
    const int rank = ctx.get_rank();

    shmem.put_nb(&rank, win.begin(), 1, right);

    shmem.barrier();

    SHTENSOR_TEST_EQUAL(win[0], left, result);

  }

  MPI_Finalize();

  return result;
}