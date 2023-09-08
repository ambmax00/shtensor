#include "Grid.h"

namespace Shtensor 
{

Grid::Grid(MPI_Comm _comm, int _grid_dim)
  : mp_cart{nullptr}
  , m_grid_dims(_grid_dim,0)
  , m_coords(_grid_dim)
{
   // create cartesian grid
  std::fill(m_grid_dims.begin(), m_grid_dims.end(), 0);

  int size = 0;
  MPI_Comm_size(_comm, &size);

  int rank = 0;
  MPI_Comm_rank(_comm, &rank);

  MPI_Dims_create(size, _grid_dim, m_grid_dims.data());

  mp_cart.reset(new MPI_Comm(MPI_COMM_NULL), 
    [](MPI_Comm* _pcomm)
    { 
      if (_pcomm && *_pcomm != MPI_COMM_NULL)
      {
        MPI_Comm_free(_pcomm);
        delete _pcomm;
      }
    }
  );

  std::vector<int> periods(_grid_dim);
  std::fill(periods.begin(), periods.end(), 1);

  MPI_Cart_create(_comm, _grid_dim, m_grid_dims.data(), periods.data(), 0, mp_cart.get());

  MPI_Cart_coords(*mp_cart, rank, _grid_dim, m_coords.data());

}

} // namespace Shtensor