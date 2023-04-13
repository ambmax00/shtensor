#include "Context.h"

namespace Shtensor 
{

Context::Context(MPI_Comm _comm, bool _attach) 
  : m_p_comm(nullptr)
  , m_p_shmem_comm(nullptr)
  , m_rank(-1)
  , m_size(0)
  , m_shmem_rank(-1)
  , m_shmem_size(0)
{
  auto comm_deleter = [](MPI_Comm* _p_comm)
  {
    if (_p_comm)
    {
      if (*_p_comm != MPI_COMM_NULL)
      {
        MPI_Comm_free(_p_comm);
      }
      delete _p_comm;
    }
  };

  if (_attach)
  {
    m_p_comm.reset(new MPI_Comm(_comm));
  }
  else 
  {
    MPI_Comm dup_comm;
    MPI_Comm_dup(_comm, &dup_comm);
    
    m_p_comm.reset(new MPI_Comm(dup_comm), comm_deleter);
  }

  MPI_Comm_rank(*m_p_comm, &m_rank);
  MPI_Comm_size(*m_p_comm, &m_size);

  m_p_shmem_comm.reset(new MPI_Comm(MPI_COMM_NULL), comm_deleter);

  MPI_Comm_split_type(*m_p_comm, MPI_COMM_TYPE_SHARED, m_rank, MPI_INFO_NULL, 
                      m_p_shmem_comm.get());
  MPI_Comm_rank(*m_p_shmem_comm, &m_shmem_rank);
  MPI_Comm_size(*m_p_shmem_comm, &m_shmem_size);
}

} // end namespace