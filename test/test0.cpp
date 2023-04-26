#include <mpi.h>
#include <stdio.h>

#include "Context.h"
#include "MemoryPool.h"
#include "Tensor.h"
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

    const std::size_t pool_memory = Shtensor::MemoryPool::MiB;

    Shtensor::MemoryPool memManager(ctx,pool_memory);

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

    std::vector<int> dim0 = {4,5,3,4};
    std::vector<int> dim1 = {5,5,8,9,2};
    std::vector<int> dim2 = {5,6,3,7,7,9};
 
    Shtensor::VArray<3> block_sizes = {dim0,dim1,dim2};

    Shtensor::Block<double,3> block({5,6,3});

    std::iota(block.begin(), block.end(), 0);

    for (int i = 0; i < block.dim(0); ++i)
    {
      for (int j = 0; j < block.dim(1); ++j)
      {
        for (int k = 0; k < block.dim(2); ++k)
        {
          printf("Block(%d,%d,%d) = %f\n", i, j, k, block(i,j,k));
        }
      }
    }

    std::array<std::size_t,5> sizes5 = {10,5,85,100,12};
    auto strides5 = Shtensor::Utils::compute_strides(sizes5);

    std::size_t long_idx = Shtensor::Utils::roll_indices(strides5, 5, 2, 75, 25, 3);

    printf("Long index: %ld\n", long_idx);

    std::array<int,5> indices5;

    Shtensor::Utils::unroll_index(strides5, long_idx, indices5);

    printf("Short idx: %d, %d, %d, %d, %d\n", indices5[0], indices5[1], indices5[2], indices5[3], indices5[4]);

    auto tensor0 = Shtensor::Tensor<double,3>(ctx, block_sizes, memManager);

    const std::vector<int> idx0 = {0,0,0,1,2,2,3,3,3};
    const std::vector<int> idx1 = {0,0,1,0,2,3,2,3,4};
    const std::vector<int> idx2 = {0,1,1,0,1,2,0,1,2};

    Shtensor::VArray<3> indices = {idx0, idx1, idx2};

    tensor0.reserve(indices);

    tensor0.print_info();

    memManager.print_info();

  }

  MPI_Finalize();

  return result;
}