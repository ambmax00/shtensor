#ifndef SHTENSOR_TENSOR_H
#define SHTENSOR_TENSOR_H

#include <array>
#include <exception>
#include <vector>

#include "Context.h"
#include "MemoryPool.h"

namespace Shtensor 
{

template <int N>
using VArray = std::array<std::vector<int>,N>;

template <int N, typename T>
class Shtensor 
{

  static_assert(N >= 1, "Tensor has to have dimension >= 1");

 public: 
  
  Shtensor(const Context& _ctx, const VArray<N>& _block_sizes) noexcept
    : m_ctx(_ctx) 
    , m_block_sizes(_block_sizes) 
  {
  }

  void reserve(const VArray& _block_idx)
  {

  }

 private: 

  const Context& m_ctx;

  const VArray<N>& m_block_sizes;
  
};


}

#endif