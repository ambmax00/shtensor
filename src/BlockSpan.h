#ifndef SHTENSOR_BLOCKSPAN_H
#define SHTENSOR_BLOCKSPAN_H

#include "Utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>

namespace Shtensor 
{

template <typename T>
class Span
{
 public:

  using iterator = T*;

  using const_iterator = T const*;

  Span()
    : m_p_data(nullptr)
    , m_size(0)
  {
  }

  Span(T* _p_data, int64_t _size)
    : m_p_data(_p_data)
    , m_size(_size)
  {
  }

  Span(const Span& _span) = default;

  Span(Span&& _span) = default;

  Span& operator=(const Span& _span) = default;

  Span& operator=(Span&& _span) = default;

  ~Span() {}

  iterator begin() { return m_p_data; }

  iterator end() { return m_p_data + m_size; }

  const_iterator begin() const { return m_p_data; }

  const_iterator end() const { return m_p_data + m_size; }

  std::size_t size() const { return static_cast<std::size_t>(m_size); }

  int64_t ssize() const { return m_size; }

  T& operator[](int64_t _idx)
  {
    return m_p_data[_idx];
  }

  const T& operator[](int64_t _idx) const
  {
    return m_p_data[_idx];
  }

  void swap(T* _pointer)
  {
    m_p_data = _pointer;
  }

 protected: 

  T* m_p_data; 

  int64_t m_size;

};

enum class BlockNorm
{
  FROBENIUS = 0,
  MAXABS = 1
};

template <typename T, int N>
class BlockSpan : public Span<T>
{
 public:

  using iterator = T*;

  using const_iterator = T const*;

  BlockSpan()
    : Span<T>(nullptr, 0)
    , m_dims({})
    , m_strides({})
  {
  }

  BlockSpan(T* _p_data, const std::array<int,N>& _dims)
    : Span<T>(_p_data, std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int64_t>()))
    , m_dims(_dims)
    , m_strides(Utils::compute_strides(_dims))
  {
  }

  BlockSpan(const BlockSpan& _span) = default;

  BlockSpan(BlockSpan&& _span) = default;

  BlockSpan& operator=(const BlockSpan& _span) = default;

  BlockSpan& operator=(BlockSpan&& _span) = default;

  ~BlockSpan() {}

  int dim(int i) const { return m_dims[i]; }

  template <class... Idx>
  T& operator()(Idx... _idx)
  {
    return this->m_p_data[Utils::roll_indices(m_strides, std::forward<Idx>(_idx)...)];
  }

  template <class... Idx>
  const T& operator()(Idx... _idx) const 
  {
    return this->m_p_data[Utils::roll_indices(m_strides, std::forward<Idx>(_idx)...)];
  }

  T get_norm(BlockNorm _norm) const 
  {
    switch (_norm)
    {
      case BlockNorm::FROBENIUS:
      {
        return std::sqrt(
          std::accumulate(this->begin(), this->end(), 0, 
            [](T sum, T val) { return (sum + std::pow(val,2)); })
          );
      }
      case BlockNorm::MAXABS:
      {
        return std::abs(
          *std::max_element(this->begin(), this->end(), 
            [](T a, T b) { return std::abs(a) < std::abs(b); })
          );
      }
      default:
      {
        return 0;
      }
    }
    
  }

 private: 

  std::array<int,N> m_dims;

  std::array<int,N> m_strides;

};


template <typename T, int N>
class Block : public BlockSpan<T,N>
{
 public:
  
  Block(const std::array<int,N>& _block_sizes) 
    : BlockSpan<T,N>(nullptr,_block_sizes)
    , m_p_storage(nullptr)
  {
    m_p_storage.reset(new T[this->size()]);
    this->swap(m_p_storage.get());
  }

  Block(const Block& _block)
    : BlockSpan<T,N>(_block)
    , m_p_storage(nullptr)
  {
    m_p_storage.reset(new T[this->size()]);
    std::copy(_block.begin(), _block.end(), m_p_storage.get());
    this->swap(m_p_storage.get());
  }

  Block(const BlockSpan<T,N>& _block_span)
    : BlockSpan<T,N>(_block_span)
    , m_p_storage(nullptr)
  {
    m_p_storage.reset(new T[this->size()]);
    std::copy(_block_span.begin(), _block_span.end(), m_p_storage.get());
    this->swap(m_p_storage.get());
  }

  Block(Block&& _block) = default;

  Block& operator=(const Block& _block)
  {
    if (&_block == this) return *this;

    BlockSpan<T,N>::operator=(_block);
    m_p_storage.reset(new T[this->size()]);
    std::copy(_block.begin(), _block.end(), m_p_storage.get());
    this->swap(m_p_storage.get());
    return *this;
  }

  ~Block() {}

 private:

  std::unique_ptr<T[]> m_p_storage;

};

} // end namespace

#endif 