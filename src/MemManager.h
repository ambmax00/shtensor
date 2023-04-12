#ifndef SHTENSOR_MEMMANAGER_H
#define SHTENSOR_MEMMANAGER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <stdio.h>

#include "Context.h"

namespace Shtensor 
{

class MemManager 
{
 public: 

  constexpr static inline std::size_t KB = 1024;
  constexpr static inline std::size_t MB = 1024*1024;
  constexpr static inline std::size_t GB = 1024*1024*1024;

  // padd chunk so we are guaranteed any type in the chunk is aligned
  struct alignas(alignof(std::max_align_t)) Chunk
  {
    Chunk* next;
    Chunk* prev;

    std::size_t data_size;
    bool free;

    std::size_t get_chunk_size() const
    {
      return sizeof(Chunk)+data_size;
    }
  };

  MemManager(const Context& _ctx, std::size_t _max_size)
    : m_ctx(_ctx)
    , m_max_size(_max_size)
    , m_p_data(nullptr)
    , m_free_mem(_max_size)
    , m_shmem_window(MPI_WIN_NULL)
    , m_window(MPI_WIN_NULL)
  {
    MPI_Win_allocate_shared(_max_size, 1, MPI_INFO_NULL, m_ctx.get_shmem_comm(), &m_p_data, 
                            &m_shmem_window);

    MPI_Win_create_dynamic(MPI_INFO_NULL, m_ctx.get_comm(), &m_window);

    MPI_Win_attach(m_window, m_p_data, m_max_size);

    Chunk* p_start_chunk = reinterpret_cast<Chunk*>(m_p_data);
    p_start_chunk->free = true;
    p_start_chunk->prev = nullptr;
    p_start_chunk->next = nullptr;
    p_start_chunk->data_size = m_max_size-sizeof(Chunk);

    m_free_mem -= sizeof(Chunk);

  }

  ~MemManager()
  {
    release();
  }

  template <typename T>
  T* allocate(std::size_t _size)
  {
    // compute byte size and round to next multiple of max alignment
    const std::size_t byte_size = _size*sizeof(T);
    const std::size_t alignment = alignof(std::max_align_t);
    const std::size_t real_data_size = (byte_size + alignment - 1) / alignment * alignment;
    const std::size_t real_chunk_size = real_data_size + sizeof(Chunk);

    // look for free chunk
    Chunk* chunk = reinterpret_cast<Chunk*>(m_p_data);
    while (chunk && !chunk->free && chunk->data_size < real_chunk_size)
    {
      chunk = chunk->next;
    }

    if (!chunk) return nullptr;

    // create new free chunk
    Chunk* new_chunk = reinterpret_cast<Chunk*>((uint8_t*)chunk+real_chunk_size);
    new_chunk->prev = chunk;
    new_chunk->next = chunk->next;
    new_chunk->data_size = chunk->data_size-real_chunk_size;
    new_chunk->free = true;

    m_free_mem -= sizeof(Chunk);

    // update old chunk
    chunk->free = false;
    chunk->data_size = real_data_size;
    chunk->next = new_chunk;

    m_free_mem -= real_data_size;

    return reinterpret_cast<T*>((uint8_t*)chunk+sizeof(Chunk));

  }

  template <typename T>
  void free(T* _p_array)
  {
    if (!_p_array) return;

    Chunk* p_chunk = reinterpret_cast<Chunk*>((uint8_t*)_p_array - sizeof(Chunk));
    p_chunk->free = true;

    m_free_mem += p_chunk->data_size;

    // merge adjacent chunks
    if (p_chunk->next != nullptr && p_chunk->next->free) 
    {
      Chunk* p_chunk_next = p_chunk->next;
      p_chunk->next = p_chunk_next->next;
      p_chunk->data_size += p_chunk_next->get_chunk_size();
      m_free_mem += sizeof(Chunk);
    }

    if (p_chunk->prev != nullptr && p_chunk->prev->free)
    {
      Chunk* p_chunk_prev = p_chunk->prev;
      p_chunk_prev->next = p_chunk->next;
      p_chunk_prev->data_size += p_chunk->get_chunk_size();
      m_free_mem += sizeof(Chunk);
    }
  }

  template <typename T>
  std::size_t get_offset(T* _p_array) const 
  {
    if (!_p_array) return 0;

    std::size_t offset = reinterpret_cast<uint8_t*>(_p_array) - m_p_data;
    
    return offset;
  }

  uint8_t* get_data() const
  {
    return m_p_data;
  }

  void print_info() const
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

  void release()
  {
    if (m_window != MPI_WIN_NULL && m_p_data)
    {
      MPI_Win_detach(m_window, m_p_data);
      MPI_Win_free(&m_window);
    }

    if (m_shmem_window != MPI_WIN_NULL)
    {
      MPI_Win_free(&m_shmem_window);
    } 

    m_window = MPI_WIN_NULL;
    m_p_data = nullptr;
    m_shmem_window = MPI_WIN_NULL;
  }

  std::size_t get_free_mem() const
  {
    return m_free_mem;
  }

 private:

  const Context& m_ctx;

  const std::size_t m_max_size;

  uint8_t* m_p_data;

  std::size_t m_free_mem;

  MPI_Win m_shmem_window;

  MPI_Win m_window;

};

} // end namespace shtensor

#endif