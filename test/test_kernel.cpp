#include <mpi.h>
#include <Kernel.h>
#include <Logger.h>

int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);
  int rank = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int result = 0;

  if (rank == 0)
  {
    
    // ijk, kim -> mj
    const std::array<int,3> sizes_ijk = {5,6,4};
    const std::array<int,3> sizes_kim = {4,5,8};
    const std::array<int,3> sizes_mj = {8,6};

    Shtensor::Kernel<double> kernel("ijk, kim -> mj", sizes_ijk, sizes_kim, sizes_mj, 1.0, 0.0);

  }

  MPI_Finalize();

  return result;
}