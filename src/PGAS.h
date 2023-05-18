#ifndef SHTENSOR_PGAS_H
#define SHTENSOR_PGAS_H

#include "Context.h"
#include "Window.h"

namespace Shtensor 
{

namespace shmem 
{

template <typename T>
Window<T> allocate(Context _ctx, int64_t _ssize)
{
  MemoryPool::unique_ptr<T> ptr = _ctx.get_mempool()->allocate_unique<T>(_ssize);
  return Window<T>(std::move(ptr),_ssize);
} 

template <typename T>
inline static void put_nb(const Context& _ctx, const T* _p_local, T* _p_remote, int64_t _ssize, 
                          int _rank)
{
  const int64_t offset = _ctx.get_mempool()->get_offset(_p_remote);
  MPI_Datatype datatype = get_mpi_type<T>();
  const int ssize32 = static_cast<int>(_ssize);

  MPI_Put(_p_local, ssize32, datatype, _rank, offset, ssize32, datatype, 
          _ctx.get_mempool()->get_window());
}

template <typename T>
inline static void put(const Context& _ctx, const T* _p_local, T* _p_remote, int64_t _ssize, 
                       int _rank)
{
  put_nb(_ctx, _p_local, _p_remote, _ssize, _rank);
  MPI_Win_flush_local(_rank, _ctx.get_mempool()->get_window());
}

inline static void quiet(const Context& _ctx)
{
  MPI_Win_flush_all(_ctx.get_mempool()->get_window());
  MPI_Win_sync(_ctx.get_mempool()->get_window());
}

inline static void barrier(const Context& _ctx)
{
  MPI_Win_flush_all(_ctx.get_mempool()->get_window());
  MPI_Win_sync(_ctx.get_mempool()->get_window());

  MPI_Request request;
  MPI_Ibarrier(_ctx.get_comm(), &request);

  int flag = 0;
  MPI_Status status;

  while (!flag)
  {
    MPI_Test(&request,&flag,&status);
  }
}


}

} // end namespace shtensor

#endif // SHTENSOR_PGAS_H