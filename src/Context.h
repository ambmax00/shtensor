#ifndef SHTENSOR_CONTEXT_HPP
#define SHTENSOR_CONTEXT_HPP

#include <mpi.h>

namespace Shtensor 
{

class Context 
{
 public: 
  
  Context(MPI_Comm _comm) 
  {
    MPI_Comm_dup(_comm, &m_comm);
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    MPI_Comm_split_type(m_comm, MPI_COMM_TYPE_SHARED, m_rank, MPI_INFO_NULL, &m_shmem_comm);
    MPI_Comm_rank(m_shmem_comm, &m_shmem_rank);
    MPI_Comm_size(m_shmem_comm, &m_shmem_size);
  }

  int get_rank() const { return m_rank; }

  int get_size() const { return m_size; }

  MPI_Comm get_comm() const { return m_comm; }

  int get_shmem_rank() const { return m_shmem_rank; }

  int get_shmem_size() const { return m_shmem_size; }

  MPI_Comm get_shmem_comm() const { return m_shmem_comm; }

 private:

  MPI_Comm m_comm;

  MPI_Comm m_shmem_comm;

  int m_rank;

  int m_size;

  int m_shmem_rank;

  int m_shmem_size;

};


} // end namespace shtensor

#endif