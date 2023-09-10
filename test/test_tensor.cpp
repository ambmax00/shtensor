#include "Context.h"
#include "MemoryPool.h"
#include "Tensor.h"
#include "TestUtils.h"

int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);

  int result = 0;

  {
    const std::size_t pool_memory = Shtensor::MiB;

    Shtensor::Context ctx(MPI_COMM_WORLD, pool_memory);

    Shtensor::Log::print(logger, "Hello from process {}\n", ctx.get_rank());

    // =============== TEST INDEX CALCULATION ===================

    const std::array<int,5> index5 = {5,2,75,25,3};
    const std::array<int,5> sizes5 = {10,5,85,100,12};

    auto strides5 = Shtensor::Utils::compute_strides(sizes5);

    int64_t long_idx5 = Shtensor::Utils::roll_indices(strides5, index5[0], index5[1], index5[2],
                                                      index5[3], index5[4]);

    std::array<int,5> index5_calc;

    Shtensor::Utils::unroll_index(strides5, long_idx5, index5_calc);

    SHTENSOR_TEST_CONTAINER_EQUAL(index5, index5_calc, result);

    // =============== TEST TENSOR CONSTRUCTION AND BLOCK ITERATION ==============

    std::vector<int> dim0 = {4,5,3,4};
    std::vector<int> dim1 = {5,5,8,9,2};
    std::vector<int> dim2 = {5,6,3,7,7,9};

    Shtensor::VArray<3> block_sizes = {dim0, dim1, dim2};

    const std::vector<int> idx0 = {0,0,0,1,2,2,3,3,3};
    const std::vector<int> idx1 = {0,0,1,0,2,3,2,3,4};
    const std::vector<int> idx2 = {0,1,1,0,1,2,0,1,2};

    Shtensor::VArray<3> indices = {idx0, idx1, idx2};

    auto tensor3 = Shtensor::Tensor<double,3>(ctx, block_sizes);
    tensor3.reserve_all();

    std::vector<int> dim3 = {4,3,2};

    Shtensor::VArray<4> block_sizes4 = {dim0,dim1,dim2,dim3};

    auto tensor4 = Shtensor::Tensor<float,4>(ctx, block_sizes4);
    tensor4.reserve_all();

    SHTENSOR_DO_BY_RANK(ctx, (tensor4.print_info()));

    SHTENSOR_TEST_EQUAL(tensor4.get_nb_nzblocks_global(), 360, result);

    for (auto iter = tensor4.begin(); iter != tensor4.end(); ++iter)
    {
      auto indices = iter.get_indices();

      SHTENSOR_TEST_EQUAL(block_sizes4[0][indices[0]], iter->dim(0), result);
      SHTENSOR_TEST_EQUAL(block_sizes4[1][indices[1]], iter->dim(1), result);
      SHTENSOR_TEST_EQUAL(block_sizes4[2][indices[2]], iter->dim(2), result);
      SHTENSOR_TEST_EQUAL(block_sizes4[3][indices[3]], iter->dim(3), result);
    }

    // ============= TEST TENSOR FILTERING ====================

    // fill every block with 0s and 1s alternatively then filter
    

    for (auto iter = tensor4.begin(); iter != tensor4.end(); ++iter)
    {
      const int64_t iblk = iter - tensor4.begin();
      const int64_t blkid = iter.get_block_index();

      const float val = (blkid % 2 == 0) ? 0.f : 1.f;

      auto block = tensor4.get_local_block(iblk);

      std::fill(block.begin(), block.end(), val);
    }

    tensor4.filter(1e-6, Shtensor::BlockNorm::FROBENIUS, false);

    // check block indices
    int64_t nb_blocks = 0;

    for (auto iter = tensor4.begin(); iter != tensor4.end(); ++iter)
    { 
      const int64_t iblk = iter - tensor4.begin();
      const int64_t blkid = iter.get_block_index();
      const int64_t ablkid = (blkid >= 0) ? blkid : -(blkid+1);

      const bool expected_empty = (ablkid % 2 == 0);
      const bool is_empty = (blkid < 0);

      nb_blocks += (is_empty) ? 0 : 1;

      SHTENSOR_TEST_EQUAL(expected_empty, is_empty, result);
    }

    tensor4.compress();

    for (auto iter = tensor4.begin(); iter != tensor4.end(); ++iter)
    { 
      const bool is_empty = (iter.get_block_index() < 0);

      SHTENSOR_TEST_TRUE(!is_empty, result);
    }

    SHTENSOR_TEST_EQUAL(nb_blocks, tensor4.get_nb_nzblocks_local(), result);

  }

  MPI_Finalize();

  return result;
}