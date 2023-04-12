#include <mpi.h>
#include <stdio.h>

#include "Context.h"
#include "MemManager.h"
#include "TestUtils.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);

  int result = 0;

  {

    char host[MPI_MAX_PROCESSOR_NAME];
    int len = -1;

    MPI_Get_processor_name(host, &len);

    Shtensor::Context ctx(MPI_COMM_WORLD);

    printf("Hello from process %d on host %s\n", ctx.get_rank(), host);

    const std::size_t pool_memory = Shtensor::MemManager::KB;

    Shtensor::MemManager memManager(ctx,pool_memory);

    const std::size_t nb_elements = 16;

    float* p_array = memManager.allocate<float>(nb_elements);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    const std::size_t comp_mem = pool_memory - nb_elements*sizeof(float) 
                                 - 2*sizeof(Shtensor::MemManager::Chunk);

    SHTENSOR_TEST_EQUAL(memManager.get_free_mem(), comp_mem, result);

    double* p_array_d = memManager.allocate<double>(3*nb_elements);

    memManager.free(p_array);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    memManager.free(p_array_d);

    SHTENSOR_DO_BY_RANK(ctx, (memManager.print_info()));

    SHTENSOR_TEST_EQUAL(memManager.get_free_mem(), 
                        pool_memory-sizeof(Shtensor::MemManager::Chunk), result);

  }

  MPI_Finalize();

  return result;
}