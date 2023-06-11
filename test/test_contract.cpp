#include "Context.h"
#include "Tensor.h"
#include "TestUtils.h"

int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);

  int result = 0;

  {
    const std::size_t pool_memory = 100*Shtensor::MiB;

    Shtensor::Context ctx(MPI_COMM_WORLD, pool_memory);

  }

  MPI_Finalize();

  return result;
}