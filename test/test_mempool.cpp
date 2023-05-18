#include "Context.h"
#include "MemoryPool.h"
#include "TestUtils.h"
#include "Logger.h"

int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);

  int result = 0;

  {
    char host[MPI_MAX_PROCESSOR_NAME];
    int len = -1;

    MPI_Get_processor_name(host, &len);

    const int64_t pool_memory = Shtensor::MiB;

    Shtensor::Context ctx(MPI_COMM_WORLD, pool_memory);

    Shtensor::Log::info(logger, "Hello from process {} on host {}", ctx.get_rank(), host);

    Shtensor::MemoryPool memManager(ctx.get_comm(), pool_memory);

    const std::size_t nb_elements = 16;

    float* p_array = memManager.allocate<float>(nb_elements);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    const std::size_t comp_mem = pool_memory - nb_elements*sizeof(float) 
                                 - 2*sizeof(Shtensor::MemoryPool::Chunk);

    SHTENSOR_TEST_EQUAL(memManager.get_free_mem(), comp_mem, result);

    double* p_array_d = memManager.allocate<double>(3*nb_elements);

    memManager.free(p_array);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    memManager.free(p_array_d);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    SHTENSOR_TEST_EQUAL(memManager.get_free_mem(), 
                        pool_memory-sizeof(Shtensor::MemoryPool::Chunk), result);
  }

  MPI_Finalize();

  return result;
}