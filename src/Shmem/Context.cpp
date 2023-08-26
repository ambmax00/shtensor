#include "Context.h"

#include <algorithm>
#include <thread>

namespace Shtensor 
{

Context::Context(MPI_Comm _comm, int64_t _vm_size, bool _attach) 
  : m_p_comm(nullptr)
  , m_p_shmem_comm(nullptr)
  , m_rank(-1)
  , m_size(0)
  , m_shmem_rank(-1)
  , m_shmem_size(0)
  , m_shmem_group_ranks(0)
  , m_p_mempool(nullptr)
  , m_nb_threads(static_cast<int>(std::thread::hardware_concurrency()))
{
  if (_attach)
  {
    m_p_comm.reset(new MPI_Comm(_comm));
  }
  else 
  {
    MPI_Comm dup_comm;
    MPI_Comm_dup(_comm, &dup_comm);
    
    m_p_comm.reset(new MPI_Comm(dup_comm), s_comm_deleter);
  }

  MPI_Comm_rank(*m_p_comm, &m_rank);
  MPI_Comm_size(*m_p_comm, &m_size);

  m_p_shmem_comm.reset(new MPI_Comm(MPI_COMM_NULL), s_comm_deleter);

  MPI_Comm_split_type(*m_p_comm, MPI_COMM_TYPE_SHARED, m_rank, MPI_INFO_NULL, 
                      m_p_shmem_comm.get());
  MPI_Comm_rank(*m_p_shmem_comm, &m_shmem_rank);
  MPI_Comm_size(*m_p_shmem_comm, &m_shmem_size);

  // communicate global rank numbers to ranks in shmem_comm
  m_shmem_group_ranks.resize(m_shmem_size,-1);

  MPI_Allgather(&m_rank, 1, MPI_INT, m_shmem_group_ranks.data(), 1, MPI_INT, *m_p_shmem_comm);

  m_p_mempool = std::make_shared<MemoryPool>(*m_p_comm, _vm_size);

}

int Context::global_to_shmem(int _rank) const
{
  auto iter = std::find(m_shmem_group_ranks.begin(), m_shmem_group_ranks.end(), _rank);
  return (iter == m_shmem_group_ranks.end()) 
          ? -1 : static_cast<int>(iter-m_shmem_group_ranks.begin());
} 

int Context::get_left_neighbour() const
{
  return (m_rank == 0) ? m_size-1 : m_rank-1;
}

int Context::get_right_neighbour() const
{
  return (m_rank == m_size-1) ? 0 : m_rank+1;
}

std::string Context::get_host_name() const 
{
  char c_name[MPI_MAX_PROCESSOR_NAME];
  int len = 0;

  MPI_Get_processor_name(c_name, &len);

  return std::string(c_name, len);
}

} // end namespace