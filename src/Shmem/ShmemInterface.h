#ifndef SHTENSOR_SHMEMINTERFACE_H
#define SHTENSOR_SHMEMINTERFACE_H

#include "BlockSpan.h"
#include "Context.h"
#include "MemoryPool.h"
#include "Window.h"

namespace Shtensor 
{

class ShmemInterface
{
 public:

  template <typename T>
  static inline constexpr MPI_Datatype get_mpi_type()
  {
    #define EXPR(ctype, mtype) \
      if (std::is_same<T,ctype>::value)\
      {\
        return mtype;\
      }

    EXPR(char, MPI_CHAR)
    EXPR(unsigned char, MPI_UNSIGNED_CHAR)
    EXPR(wchar_t, MPI_WCHAR)
    EXPR(short, MPI_SHORT)
    EXPR(unsigned short, MPI_UNSIGNED_SHORT)
    EXPR(int, MPI_INT)
    EXPR(unsigned, MPI_UNSIGNED)
    EXPR(long, MPI_LONG)
    EXPR(unsigned long, MPI_UNSIGNED_LONG)
    EXPR(long long, MPI_LONG_LONG)
    EXPR(unsigned long long, MPI_UNSIGNED_LONG_LONG)
    EXPR(float, MPI_FLOAT)
    EXPR(double, MPI_DOUBLE)
    EXPR(long double, MPI_LONG_DOUBLE)
    EXPR(int8_t, MPI_INT8_T)
    EXPR(int16_t, MPI_INT16_T)
    EXPR(int32_t, MPI_INT32_T)
    EXPR(int64_t, MPI_INT64_T)
    EXPR(uint8_t, MPI_UINT8_T)
    EXPR(uint16_t, MPI_UINT16_T)
    EXPR(uint32_t, MPI_UINT32_T)
    EXPR(uint64_t, MPI_UINT64_T)
    EXPR(bool, MPI_C_BOOL)

    static_assert(MPI_DATATYPE_NULL, "Unknown MPI datatype");

    return MPI_DATATYPE_NULL;

    #undef EXPR
  }

  ShmemInterface(const Context& _ctx)
    : m_ctx(_ctx)
  {
  }

  template <typename T>
  Window<T> allocate(int64_t _ssize)
  {
    auto p_mempool = m_ctx.get_mempool();
    T* p_data = p_mempool->allocate<T>(_ssize);
    Window<T> win(p_data, _ssize, p_mempool->get_deleter<T>(), p_mempool->get_reallocator<T>());
    return win;
  } 

  template <typename T>
  void put_nb(const T* _p_local, T* _p_remote, int64_t _ssize, int _rank)
  {
    const int64_t offset = m_ctx.get_mempool()->get_offset(_p_remote);
    MPI_Datatype datatype = get_mpi_type<T>();
    const int ssize32 = static_cast<int>(_ssize);

    MPI_Put(_p_local, ssize32, datatype, _rank, offset, ssize32, datatype, 
            m_ctx.get_mempool()->get_window());
  }

  template <typename T>
  void put(const T* _p_local, T* _p_remote, int64_t _ssize, int _rank)
  {
    put_nb(m_ctx, _p_local, _p_remote, _ssize, _rank);
    MPI_Win_flush_local(_rank, m_ctx.get_mempool()->get_window());
  }

  void quiet()
  {
    MPI_Win_flush_all(m_ctx.get_mempool()->get_window());
    MPI_Win_sync(m_ctx.get_mempool()->get_window());
  }

  void barrier()
  {
    MPI_Win_flush_all(m_ctx.get_mempool()->get_window());
    MPI_Win_sync(m_ctx.get_mempool()->get_window());

    MPI_Request request;
    MPI_Ibarrier(m_ctx.get_comm(), &request);

    int flag = 0;
    MPI_Status status;

    while (!flag)
    {
      MPI_Test(&request,&flag,&status);
    }
  }


 private:

  Context m_ctx;

};

} // end namespace shtensor

#endif // SHTENSOR_SHMEMINTERFACE_H