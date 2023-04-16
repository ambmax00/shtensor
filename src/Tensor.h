#ifndef SHTENSOR_TENSOR_H
#define SHTENSOR_TENSOR_H

#include <array>
#include <exception>
#include <numeric>
#include <vector>

#include "BlockSpan.h"
#include "Context.h"
#include "MemoryPool.h"

namespace Shtensor 
{

template <int N>
using VArray = std::array<std::vector<int>,N>;

template <int N, typename T>
class Tensor 
{

  static_assert(N >= 1, "Tensor has to have dimension >= 1");

 public: 
  
  Tensor(const Context& _ctx, const VArray<N>& _block_sizes, MemoryPool& _mempool) noexcept
    : m_ctx(_ctx) 
    , m_block_sizes(_block_sizes) 
    , m_pool(_mempool)
    , m_grid_dims({})
    , m_p_cart(nullptr)
    , m_p_data(nullptr)
    , m_p_sparse_data(nullptr)
    , m_p_row_idx(nullptr)
    , m_p_slice_idx(nullptr)
  {
    // create cartesian grid
    /*std::fill(m_grid_dims.begin(), m_grid_dims.end(), 0);

    MPI_Dims_create(m_ctx.get_size(), N, m_grid_dims.data());

    m_p_cart.reset(new MPI_Comm(MPI_COMM_NULL), Context::s_comm_deleter);

    std::array<int,N> periods;
    std::fill(periods.begin(), periods.end(), 1);

    MPI_Cart_create(m_ctx.get_comm(), N, m_grid_dims.data(), periods.data(), 0, m_p_cart.get());

    m_p_sparse_data = m_pool.allocate<uint8_t>(m_block_sizes[0]*sizeof(int));
    m_p_row_idx = reinterpret_cast<int>(m_p_sparse_data);
    m_p_slice_idx = nullptr;*/

  }

  ~Tensor()
  {
  }

  std::size_t get_dim_size(int i)
  {
    if (i < N || i >= N) return 0;

    return std::accumulate(m_block_sizes[i].begin(), m_block_sizes[i].end(), std::size_t(0));
  }

  void reserve(const VArray<N>& _block_idx)
  {

  }

 private: 

  const Context m_ctx;

  const VArray<N> m_block_sizes;

  MemoryPool m_pool;

  std::array<int,N> m_grid_dims;

  std::shared_ptr<MPI_Comm> m_p_cart;

  T* m_p_data;

  uint8_t* m_p_sparse_data;

  int* m_p_row_idx;

  int64_t* m_p_slice_idx;
  
};


}

#endif