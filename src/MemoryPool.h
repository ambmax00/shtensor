#ifndef SHTENSOR_MEMORYPOOL_H
#define SHTENSOR_MEMORYPOOL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <stdio.h>

#include "Logger.h"
#include "Utils.h"

namespace Shtensor 
{

class MemoryPool 
{
 public: 

  // padd chunk so we are guaranteed any type in the chunk is aligned
  struct alignas(alignof(std::max_align_t)) Chunk
  {
    Chunk* next;
    Chunk* prev;

    int64_t data_size;
    bool free;

    int64_t get_chunk_size() const
    {
      return SSIZEOF(Chunk)+data_size;
    }
  };

  MemoryPool();

  MemoryPool(MPI_Comm _comm, int64_t _max_size);

  ~MemoryPool();

  MemoryPool(const MemoryPool& _pool) = delete;

  MemoryPool(MemoryPool&& _pool) = default;

  MemoryPool& operator=(const MemoryPool& _pool) = delete;

  MemoryPool& operator=(MemoryPool&& _pool) = default;

  template <typename T>
  [[nodiscard]] T* allocate(int64_t _size);

  template <typename T>
  void free(T* _p_array);

  template <typename T>
  constexpr auto get_deleter()
  {
    return [this](T* _ptr) mutable { free(_ptr); };
  }

  template <typename T>
  using deleter_function = 
    typename std::result_of<decltype(&MemoryPool::get_deleter<T>)(MemoryPool)>::type;

  template <typename T>
  using unique_ptr = std::unique_ptr<T,deleter_function<T>>;

  template <typename T> 
  std::shared_ptr<T> allocate_shared(int64_t _size);

  template <typename T>
  unique_ptr<T> allocate_unique(int64_t);

  template <typename T>
  int64_t get_offset(T* _p_array) const;

  template <typename T>
  int64_t get_size(T* _p_array) const;

  template <typename T>
  int64_t find_offset(const T* _ptr) const;

  uint8_t* get_data() const { return m_p_data; }

  void print_info() const;

  void release();

  int64_t get_free_mem() const { return m_free_mem; }

  const MPI_Comm& get_comm() const { return m_comm; }

  MPI_Win get_window() const { return *m_p_window; }

  //MPI_Win get_shmem_window() const { return *m_p_shmem_window; }

  //uint8_t* get_shmem_begin(int shmem_rank) const;

 private:

  MPI_Comm m_comm;

  int64_t m_max_size;

  uint8_t* m_p_data;

  int64_t m_free_mem;

  std::shared_ptr<MPI_Win> m_p_shmem_window;

  std::shared_ptr<MPI_Win> m_p_window;

  Log::Logger m_logger;

};

template <typename T>
T* MemoryPool::allocate(int64_t _size)
{
  // compute byte size and round to next multiple of max alignment
  const int64_t byte_size = _size*SSIZEOF(T);
  const int64_t alignment = alignof(std::max_align_t);
  const int64_t real_data_size = (byte_size + alignment - 1) / alignment * alignment;
  const int64_t real_chunk_size = real_data_size + SSIZEOF(Chunk);

  // look for free chunk
  Chunk* chunk = reinterpret_cast<Chunk*>(m_p_data);
  while (chunk) 
  {
    if (chunk->free && chunk->data_size >= real_data_size)
    {
      break;
    }
    chunk = chunk->next;
  }

  if (!chunk) 
  {
    Log::error(m_logger, "Memorypool is out of memory");
    return nullptr;
  }

  // create new free chunk
  Chunk* new_chunk = reinterpret_cast<Chunk*>((uint8_t*)chunk+real_chunk_size);
  new_chunk->prev = chunk;
  new_chunk->next = chunk->next;
  new_chunk->data_size = chunk->data_size-real_chunk_size;
  new_chunk->free = true;

  m_free_mem -= SSIZEOF(Chunk);

  // update old chunk
  chunk->free = false;
  chunk->data_size = real_data_size;
  chunk->next = new_chunk;

  m_free_mem -= real_data_size;

  return reinterpret_cast<T*>((uint8_t*)chunk+SSIZEOF(Chunk));

}

template <typename T>
void MemoryPool::free(T* _p_array)
{
  if (!_p_array) return;

  Chunk* p_chunk = reinterpret_cast<Chunk*>((uint8_t*)_p_array - SSIZEOF(Chunk));
  p_chunk->free = true;

  m_free_mem += p_chunk->data_size;

  // merge adjacent chunks
  if (p_chunk->next != nullptr && p_chunk->next->free) 
  {
    Chunk* p_chunk_next = p_chunk->next;
    p_chunk->next = p_chunk_next->next;
    p_chunk->data_size += p_chunk_next->get_chunk_size();
    m_free_mem += SSIZEOF(Chunk);
  }

  if (p_chunk->prev != nullptr && p_chunk->prev->free)
  {
    Chunk* p_chunk_prev = p_chunk->prev;
    p_chunk_prev->next = p_chunk->next;
    p_chunk_prev->data_size += p_chunk->get_chunk_size();
    m_free_mem += SSIZEOF(Chunk);
  }
}

template <typename T> 
std::shared_ptr<T> MemoryPool::allocate_shared(int64_t _size)
{
  return std::shared_ptr<T>(this->allocate<T>(_size),get_deleter<T>());
}

template <typename T>
MemoryPool::unique_ptr<T> MemoryPool::allocate_unique(int64_t _size)
{
  return unique_ptr<T>(this->allocate<T>(_size),get_deleter<T>());
}

template <typename T>
int64_t MemoryPool::get_offset(T* _p_array) const 
{
  if (!_p_array) return 0;

  int64_t offset = reinterpret_cast<uint8_t*>(_p_array) - m_p_data;
  
  return offset;
}

template <typename T>
int64_t MemoryPool::get_size(T* _p_array) const
{
  if (!_p_array) return 0;

  Chunk* p_chunk = reinterpret_cast<Chunk*>(reinterpret_cast<uint8_t*>(_p_array) - SSIZEOF(Chunk));

  return p_chunk->data_size;
}

using SharedMemoryPool = std::shared_ptr<MemoryPool>;

} // end namespace shtensor

#endif