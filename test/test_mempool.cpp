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
    const int64_t chunk_ssize = SSIZEOF(Shtensor::MemoryPool::Chunk);
    const int64_t pool_memory = 1024*SSIZEOF(double)+chunk_ssize;

    Shtensor::Context ctx(MPI_COMM_WORLD, pool_memory);

    Shtensor::Log::info(logger, "Hello from process {} on host {}", ctx.get_rank(), 
                        ctx.get_host_name());

    Shtensor::MemoryPool mem_pool(ctx.get_comm(), pool_memory);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    const int64_t expected_free_mem_full = pool_memory - chunk_ssize;

    SHTENSOR_TEST_EQUAL(expected_free_mem_full, mem_pool.get_free_mem(), result);

    // test allocate and free

    double* p_darray_0 = mem_pool.allocate<double>(100);
    float* p_farray_1 = mem_pool.allocate<float>(100);

    const int64_t expected_free_mem_0 = pool_memory
                                        - 100*SSIZEOF(double) - 100*SSIZEOF(float)
                                        - 3*chunk_ssize;

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    SHTENSOR_TEST_EQUAL(expected_free_mem_0, mem_pool.get_free_mem(), result);

    mem_pool.free(p_darray_0);

    const int64_t expected_free_mem_1 = pool_memory - 100*SSIZEOF(float)
                                        - 3*chunk_ssize;

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    SHTENSOR_TEST_EQUAL(expected_free_mem_1, mem_pool.get_free_mem(), result);

    double* p_darray_1 = mem_pool.allocate<double>(100);

    // Memory pool should pick first free block again
    SHTENSOR_TEST_EQUAL((int64_t)p_darray_0, (int64_t)p_darray_1, result);

    mem_pool.free(p_darray_1);

    mem_pool.free(p_farray_1);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    // Memory pool should return to initial size
    const int64_t expected_free_mem_2 = pool_memory - chunk_ssize;

    SHTENSOR_TEST_EQUAL(expected_free_mem_2, mem_pool.get_free_mem(), result);

    uint8_t* p_uarray_0 = mem_pool.allocate<uint8_t>(pool_memory-chunk_ssize);

    p_uarray_0 = mem_pool.reallocate(p_uarray_0, 256);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    const int64_t expected_free_mem_3 = pool_memory - 256 - 2*chunk_ssize;

    SHTENSOR_TEST_EQUAL(expected_free_mem_3, mem_pool.get_free_mem(), result);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    p_uarray_0 = mem_pool.reallocate(p_uarray_0, 512);

    const int64_t expected_free_mem_4 = pool_memory - 512 - 2*chunk_ssize;

    SHTENSOR_TEST_EQUAL(expected_free_mem_4, mem_pool.get_free_mem(), result);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

    p_uarray_0 = mem_pool.reallocate(p_uarray_0, pool_memory-chunk_ssize);

    const int64_t expected_free_mem_5 = 0;

    SHTENSOR_TEST_EQUAL(expected_free_mem_5, mem_pool.get_free_mem(), result);

    if (ctx.get_rank() == 0)
    {
      mem_pool.print_info();
    }

  }

  MPI_Finalize();

  return result;
}