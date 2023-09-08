#ifndef SHTENSOR_GRID_H
#define SHTENSOR_GRID_H

#include <mpi.h>

#include <memory>
#include <vector>

#include "Utils.h"

namespace Shtensor
{

class Grid
{
 public: 

  Grid(MPI_Comm _comm, int _grid_dim);

  MPI_Comm get_cart() { return *mp_cart; }

  const std::vector<int>& get_grid_dims() { return m_grid_dims; }

  const std::vector<int>& get_coords() { return m_coords; }

  int get_dim() { return Utils::ssize(m_coords); }
 
 private:

  std::shared_ptr<MPI_Comm> mp_cart;
  std::vector<int> m_grid_dims;
  std::vector<int> m_coords;  

};

} // namespace Shtensor

#endif