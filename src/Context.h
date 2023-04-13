#ifndef SHTENSOR_CONTEXT_HPP
#define SHTENSOR_CONTEXT_HPP

#include <memory>
#include <mpi.h>

namespace Shtensor 
{

class Context 
{
 public: 
  
  Context(MPI_Comm _comm, bool _attach = false);

  Context(const Context& _ctx) = default;

  Context(Context&& _ctx) = default;

  Context& operator=(const Context& _ctx) = default; 

  Context& operator=(Context&& _ctx) = default;

  ~Context() {}

  int get_rank() const { return m_rank; }

  int get_size() const { return m_size; }

  MPI_Comm get_comm() const { return *m_p_comm; }

  int get_shmem_rank() const { return m_shmem_rank; }

  int get_shmem_size() const { return m_shmem_size; }

  MPI_Comm get_shmem_comm() const { return *m_p_shmem_comm; }

 private:

  std::shared_ptr<MPI_Comm> m_p_comm;

  std::shared_ptr<MPI_Comm> m_p_shmem_comm;

  int m_rank;

  int m_size;

  int m_shmem_rank;

  int m_shmem_size;

};


} // end namespace shtensor

#endif