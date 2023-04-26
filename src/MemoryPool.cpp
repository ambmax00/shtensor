#include "MemoryPool.h"

namespace Shtensor 
{
  
MemoryPool::MemoryPool(const Context& _ctx, int64_t _max_size)
  : m_ctx(_ctx)
  , m_max_size(_max_size)
  , m_p_data(nullptr)
  , m_free_mem(_max_size)
  , m_p_shmem_window(nullptr)
  , m_p_window(nullptr)
{

  auto window_deleter = [](MPI_Win* _p_win)
  {
    if (_p_win)
    {
      if (*_p_win != MPI_WIN_NULL)
      {
        MPI_Win_free(_p_win);
      }
      delete _p_win;
    }
  };

  m_p_shmem_window.reset(new MPI_Win(MPI_WIN_NULL), window_deleter);
  m_p_window.reset(new MPI_Win(MPI_WIN_NULL), window_deleter);

  MPI_Win_allocate_shared(_max_size, 1, MPI_INFO_NULL, m_ctx.get_shmem_comm(), &m_p_data, 
                          m_p_shmem_window.get());

  MPI_Win_create_dynamic(MPI_INFO_NULL, m_ctx.get_comm(), m_p_window.get());

  MPI_Win_attach(*m_p_window, m_p_data, m_max_size);

  Chunk* p_start_chunk = reinterpret_cast<Chunk*>(m_p_data);
  p_start_chunk->free = true;
  p_start_chunk->prev = nullptr;
  p_start_chunk->next = nullptr;
  p_start_chunk->data_size = m_max_size-SSIZEOF(Chunk);

  m_free_mem -= SSIZEOF(Chunk);

}

MemoryPool::~MemoryPool()
{
}

void MemoryPool::print_info() const
{
  printf("Memory pool info for rank %d\n", m_ctx.get_rank());
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

} // end namespace