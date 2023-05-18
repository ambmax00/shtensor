#ifndef SHTENSOR_CONTEXT_HPP
#define SHTENSOR_CONTEXT_HPP

#include <memory>
#include <mpi.h>
#include <vector>

#include "Logger.h"
#include "MemoryPool.h"

namespace Shtensor 
{

class Context 
{
 public: 
  
  static inline auto s_comm_deleter = [](MPI_Comm* _p_comm)
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

  Context(MPI_Comm _comm, int64_t _vmsize, bool _attach = false);

  Context(const Context& _ctx) = default;

  Context(Context&& _ctx) = default;

  Context& operator=(const Context& _ctx) = default; 

  Context& operator=(Context&& _ctx) = default;

  ~Context() {}

  int get_rank() const { return m_rank; }

  int get_size() const { return m_size; }

  const SharedMemoryPool& get_mempool() const { return m_p_mempool; }

  MPI_Comm get_comm() const { return *m_p_comm; }

  int get_shmem_rank() const { return m_shmem_rank; }

  int get_shmem_size() const { return m_shmem_size; }

  MPI_Comm get_shmem_comm() const { return *m_p_shmem_comm; }

  int global_to_shmem(int _rank) const;

  int get_left_neighbour() const;

  int get_right_neighbour() const;

  std::string get_host_name() const;

 private:

  std::shared_ptr<MPI_Comm> m_p_comm;

  std::shared_ptr<MPI_Comm> m_p_shmem_comm;

  int m_rank;

  int m_size;

  int m_shmem_rank;

  int m_shmem_size;

  std::vector<int> m_shmem_group_ranks;

  SharedMemoryPool m_p_mempool;

};


} // end namespace shtensor

#endif