#include "MemoryPool.h"

namespace Shtensor 
{
  
MemoryPool::MemoryPool()
  : m_comm(MPI_COMM_NULL)
  , m_max_size(0)
  , m_p_data(nullptr)
  , m_free_mem(0)
  , m_p_shmem_window(nullptr)
  , m_p_window(nullptr)
  , m_logger(Log::create("MemoryPool"))
{
}

MemoryPool::MemoryPool(MPI_Comm _comm, int64_t _max_size)
  : m_comm(_comm)
  , m_max_size(_max_size)
  , m_p_data(nullptr)
  , m_free_mem(_max_size)
  , m_p_shmem_window(nullptr)
  , m_p_window(nullptr)
  , m_logger(Log::create("MemoryPool"))
{

  // auto shmem_window_deleter = [](MPI_Win* _p_win)
  // {
  //   if (_p_win)
  //   {
  //     if (*_p_win != MPI_WIN_NULL)
  //     {
  //       MPI_Win_free(_p_win);
  //     }
  //     delete _p_win;
  //   }
  // };

  auto window_deleter = [this](MPI_Win* _pwin)
  {
    if (_pwin)
    {
      if (*_pwin != MPI_WIN_NULL)
      {
        MPI_Win_unlock_all(*_pwin);
        MPI_Win_free(_pwin);
      }
      delete _pwin;
    }
  };

  //m_p_shmem_window.reset(new MPI_Win(MPI_WIN_NULL), shmem_window_deleter);
  m_p_window.reset(new MPI_Win(MPI_WIN_NULL), window_deleter);

  // TO DO: ALLOC SHARED NON CONTIGUOUS!!

  //MPI_Win_allocate_shared(_max_size, 1, MPI_INFO_NULL, m_ctx.get_shmem_comm(), &m_p_data, 
  //                        m_p_shmem_window.get());

  //MPI_Info info;
  //MPI_Info_create(&info);
  //MPI_Info_set(info, "coll_attach", "true");

  //MPI_Win_create_dynamic(MPI_INFO_NULL, m_ctx.get_comm(), m_p_window.get());

  //MPI_Win_attach(*m_p_window, m_p_data, m_max_size);

  MPI_Win_allocate(_max_size, 1, MPI_INFO_NULL, m_comm, &m_p_data, m_p_window.get());

  /*int* win_model = nullptr;
  int has_attr = 0;
  int status = MPI_Win_get_attr(*m_p_window, MPI_WIN_MODEL, &win_model, &has_attr);

  Log::debug(m_logger, "has_attr: {}, model: {}", has_attr, *win_model);

  if (has_attr && (*win_model != MPI_WIN_UNIFIED))
  {
    std::string err_msg = "Shtensor needs unified RMA model.";
    Log::critical(m_logger, err_msg);
    throw std::runtime_error(err_msg);
  }
  else if (!has_attr)
  {
    std::string err_msg = "Shtensor could not find attribute MPI_WIN_MODEL.";
    Log::critical(m_logger, err_msg);
    throw std::runtime_error(err_msg);
  }*/

  Chunk* p_start_chunk = reinterpret_cast<Chunk*>(m_p_data);
  p_start_chunk->free = true;
  p_start_chunk->prev = nullptr;
  p_start_chunk->next = nullptr;
  p_start_chunk->data_size = m_max_size-SSIZEOF(Chunk);

  m_free_mem -= SSIZEOF(Chunk);

  MPI_Win_lock_all(MPI_MODE_NOCHECK, *m_p_window);

}

MemoryPool::~MemoryPool()
{
}

void MemoryPool::print_info() const
{
  int rank = -1;
  MPI_Comm_rank(m_comm, &rank);

  printf("Memory pool info for rank %d\n", rank);
  printf("Memory pool of size %ld\n", m_max_size);
  printf("Free memory: %ld\n", m_free_mem);

  Chunk* chunk = reinterpret_cast<Chunk*>(m_p_data);
  int i = 0;

  while (chunk)
  {
    printf("Chunk %d with size %ld is %s\n", i, 
            chunk->data_size, chunk->free ? "free" : "occupied");

    chunk = chunk->next;
    ++i;
  }
}

void MemoryPool::release()
{
  m_p_window.reset();
  m_p_shmem_window.reset();
}

// uint8_t* MemoryPool::get_shmem_begin(int shmem_rank) const
// {
//   uint8_t* p_out = nullptr;
//   MPI_Aint size = -1;
//   int disp = -1;

//   MPI_Win_shared_query(*m_p_shmem_window, shmem_rank, &size, &disp, &p_out);

//   return (size > 0 && disp > 0) ? p_out : nullptr;
// }

} // end namespace