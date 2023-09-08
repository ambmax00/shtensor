#ifndef SHTENSOR_TENSOR_H
#define SHTENSOR_TENSOR_H

#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "BaseTensor.h"
#include "BlockSpan.h"
#include "Context.h"
#include "MemoryPool.h"
#include "ShmemInterface.h"
#include "ThreadPool.h"

namespace Shtensor 
{

template <typename T, int N>
class Tensor : public BaseTensor
{

  static_assert(N >= 2, "Tensor has to have dimension >= 2");

 public: 
  
  Tensor(const Context& _ctx, const VArray<N>& _block_sizes) noexcept
   : BaseTensor(_ctx, 
                VVector<int>(_block_sizes.begin(), _block_sizes.end()), 
                N, Utils::float_type<T>())
  {
  }

  ~Tensor()
  {
  }

  BlockSpan<T,N> get_local_block(int64_t _idx)
  {
    std::array<int,N> indices;
    Utils::unroll_index(m_block_strides, N, _idx, indices);

    std::array<int,N> sizes;
    for (int i = 0; i < N; ++i)
    {
      sizes[i] = m_block_sizes[i][indices[i]];
    }

    T* p_data = reinterpret_cast<T*>(m_win_data.data()) + m_sparse_info.offsets[_idx];

    return BlockSpan<T,N>(p_data, sizes);
  }

  std::array<int,N> get_coords_block(const std::array<int,N>& _idx)
  {
    std::array<int,N> coords;
    for (int i = 0; i < N; ++i)
    {
      coords[i] = m_block_coords[i][_idx[i]];
    }
    return coords;
  }

  int get_rank_block(const std::array<int,N>& _idx)
  {
    auto coords = get_coords_block(_idx);
    int rank = -1;
    MPI_Cart_rank(m_grid.get_cart(), coords.data(), &rank);
    return rank;
  }

  void filter(T _eps, BlockNorm _norm, bool _compress = true);

#if 1
  template <class ValueType, class Pointer, class Reference>
  class BlockIteratorDetail
  {
   public:

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = int64_t;
    using value_type = ValueType;
    using pointer = Pointer;  
    using reference = Reference;

    BlockIteratorDetail(const Tensor<T,N>& _tensor, int64_t _idx)
      : m_tensor(_tensor)
      , m_idx(_idx)
      , m_block_span()
    {
      update_block();
    }

    BlockIteratorDetail(const BlockIteratorDetail& _iter) = default;

    BlockIteratorDetail(BlockIteratorDetail&& _iter) = default;

    BlockIteratorDetail& operator=(const BlockIteratorDetail& _iter) = default;

    BlockIteratorDetail& operator=(BlockIteratorDetail&& _iter) = default;

    ~BlockIteratorDetail() {}

    std::array<int,N> get_indices() const 
    { 
      std::array<int,N> out;
      Utils::unroll_index(m_tensor.m_block_strides, N, m_idx, out);
      return out;
    }

    int64_t get_block_index() const 
    {
      return m_tensor.m_sparse_info.indices[m_idx];
    }

    reference operator*()
    {
      return m_block_span;
    }

    pointer operator->()
    {
      return &m_block_span;
    }

    BlockIteratorDetail& operator++() 
    { 
      m_idx++; 
      update_block();
      return *this;
    }
    
    BlockIteratorDetail operator++(int)
    {
      BlockIteratorDetail tmp_iter(*this);
      m_idx++;
      update_block();
      return tmp_iter;
    }
    
    BlockIteratorDetail& operator+=(difference_type _dt)
    {
      m_idx += _dt;
      update_block();
      return *this;
    }
    
    BlockIteratorDetail operator+(difference_type _dt) const
    {
      BlockIteratorDetail tmp_iter(*this);
      tmp_iter += _dt;
      tmp_iter.update_block();
      return tmp_iter;
    }

    BlockIteratorDetail& operator--() 
    { 
      m_idx--; 
      update_block();
      return *this;
    }
    
    BlockIteratorDetail operator--(int)
    {
      BlockIteratorDetail tmp_iter(*this);
      m_idx--;
      update_block();
      return tmp_iter;
    }
    
    BlockIteratorDetail& operator-=(difference_type _dt)
    {
      m_idx -= _dt;
      update_block();
      return *this;
    }
    
    BlockIteratorDetail operator-(difference_type _dt) const
    {
      BlockIteratorDetail tmp_iter(*this);
      tmp_iter -= _dt;
      tmp_iter.update_block();
      return tmp_iter;
    }
    
    difference_type operator-(const BlockIteratorDetail& _iter) const
    {
      return m_idx - _iter.m_idx;
    }

    bool operator<(const BlockIteratorDetail& _iter) const 
    {
      return m_idx < _iter.m_idx;
    }

    bool operator>(const BlockIteratorDetail& _iter) const 
    {
      return m_idx > _iter.m_idx;
    }

    bool operator<=(const BlockIteratorDetail& _iter) const 
    {
      return m_idx <= _iter.m_idx;
    }

    bool operator>=(const BlockIteratorDetail& _iter) const 
    {
      return m_idx >= _iter.m_idx;
    }

    bool operator==(const BlockIteratorDetail& _iter) const 
    {
      return m_idx == _iter.m_idx;
    }

    bool operator!=(const BlockIteratorDetail& _iter) const 
    {
      return m_idx != _iter.m_idx;
    }

    reference operator[](difference_type _dt)
    {
      return *(*this + _dt);
    }

   private:

    void update_block()
    {
      // deleted block or one past end 
      if (m_idx < 0 || m_idx == m_tensor.m_sinfo_local.nb_nzblocks) 
      {
        m_block_span = BlockSpan<T,N>(nullptr, std::array<int,N>{});
        return;
      }

      std::array<int,N> indices;
      Utils::unroll_index(m_tensor.m_block_strides, N, m_idx, indices);

      std::array<int,N> sizes;
      for (int i = 0; i < N; ++i)
      {
        sizes[i] = m_tensor.m_block_sizes[i][indices[i]];
      }

      int64_t offset = m_tensor.m_sparse_info.offsets[m_idx];
      T* p_data = reinterpret_cast<T*>(const_cast<uint8_t*>(m_tensor.m_win_data.data())) + offset;

      m_block_span = BlockSpan<T,N>(p_data, sizes);
    }

    const Tensor<T,N>& m_tensor;

    difference_type m_idx;

    BlockSpan<T,N> m_block_span;

  };

  using BlockIterator = BlockIteratorDetail<Block<T,N>,BlockSpan<T,N>*,BlockSpan<T,N>&>;

  using ConstBlockIterator = BlockIteratorDetail<const Block<T,N>,const BlockSpan<T,N>*,
                                                 const BlockSpan<T,N>&>;

  BlockIterator begin()
  {
    return BlockIterator(*this,0);
  }

  BlockIterator end()
  {
    return BlockIterator(*this,m_sinfo_local.nb_nzblocks);
  }

  ConstBlockIterator cbegin() const
  {
    return ConstBlockIterator(*this,0);
  }

  ConstBlockIterator cend() const
  {
    return ConstBlockIterator(*this,m_sinfo_local.nb_nzblocks);
  }

#endif

  void print_blocks()
  {
    for (ConstBlockIterator iter = cbegin(); iter < cend(); ++iter)
    {
      auto indices = iter.get_indices();
      Log::print(m_logger, "({})[{}]\n", fmt::join(indices, ","), fmt::join(*iter, ","));
    }
  }

};

/*template <class T, int N>
class BlockView
{
 public:

  class BlockIteratorDetail 
  {
    public:
      using iterator_category = std::random_access_iterator_tag;
      using difference_type   = std::ptrdiff_t;
      using value_type        = BlockSpan<T,N>;
      using pointer           = BlockSpan<T,N>*;  // or also value_type*
      using reference         = BlockSpan<T,N>&;  // or also value_type&

      BlockIteratorDetail(Tensor<T,N>& _tensor, std::size_t _idx)
        : m_tensor(_tensor)
        , m_block_idx(_idx)
        , m_indices()
      {

      }

    private:

      std::size_t m_block_idx;

      std::array<int,N> m_indices;
  };

 private:

  Tensor<N,T> m_tensor;

};*/

template <typename T, int N>
using SharedTensor = std::shared_ptr<Tensor<T,N>>;

template <typename T, int N>
void Tensor<T,N>::filter(T _eps, BlockNorm _norm, bool _compress)
{
  {
    ThreadPool tpool(2*m_ctx.get_nb_threads());

    int64_t start = 0;
    int64_t end = m_sinfo_local.nb_nzblocks;
    int64_t step = 1;

    auto loop_func = [this,_eps,_norm](int64_t _id)
    {
      auto iter = this->begin() + _id;
      if (iter->get_norm(_norm) < _eps) 
      {
        this->m_sparse_info.indices[_id] = -this->m_sparse_info.indices[_id]-1;
      }
    };

    tpool.run(start,end,step,loop_func);
  }

  if (_compress)
  {
    compress();
  }

}

}

#endif