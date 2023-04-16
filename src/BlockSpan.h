#ifndef SHTENSOR_BLOCKSPAN_H
#define SHTENSOR_BLOCKSPAN_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>

namespace Shtensor 
{

template <int N, typename T>
class BlockSpan
{
 public:

  using iterator = T*;

  using const_iterator = T const*;

  template <int M = N, typename std::enable_if<M == 1,int>::type = 0>
  BlockSpan(T* _p_data, std::size_t _size)
    : m_p_data(_p_data)
    , m_dims({_size})
    , m_strides({1})
    , m_size(_size)
  {
  }

  BlockSpan(T* _p_data, const std::array<std::size_t,N>& _dims)
    : m_p_data(_p_data)
    , m_dims(_dims)
    , m_strides({})
    , m_size(std::accumulate(_dims.begin(), _dims.end(), 1ul, std::multiplies<std::size_t>()))
  {
    std::generate(m_strides.begin(), m_strides.end(), 
      [dims=m_dims,i=0]() mutable
      {
        std::size_t stride = std::accumulate(dims.begin(), dims.begin()+i, 1.0, 
                                             std::multiplies<std::size_t>());
        ++i;
        return stride;
      });
  }

  BlockSpan(const BlockSpan& _span) = default;

  BlockSpan(BlockSpan&& _span) = default;

  BlockSpan& operator=(const BlockSpan& _span) = default;

  BlockSpan& operator=(BlockSpan&& _span) = default;

  ~BlockSpan() {}

  iterator begin() { return m_p_data; }

  iterator end() { return m_p_data + m_size; }

  const_iterator begin() const { return m_p_data; }

  const_iterator end() const { return m_p_data + m_size; }

  std::size_t size() const { return m_size; }

  std::size_t dim(int i) const { return m_dims[i]; }

  template <class... Idx>
  T& operator()(Idx... _idx)
  {
    return access<Idx...>(std::forward<Idx>(_idx)..., std::make_index_sequence<N>{});
  }

  template <class... Idx>
  const T& operator()(Idx... _idx) const 
  {
    return access(_idx..., std::make_index_sequence<N>{});
  }

 protected:

  void swap(T* _pointer)
  {
    m_p_data = _pointer;
  }

 private: 

  template <typename I>
  static inline constexpr bool is_valid_index()
  {
    return std::is_same<I,std::size_t>::value 
            || std::is_same<I,int>::value 
            || std::is_same<I,int64_t>::value;
  }

  template <class... Idx, std::size_t... Is> 
  inline T& access(Idx... _idx, std::index_sequence<Is...> const &)
  {
    static_assert((is_valid_index<Idx>() && ...), "Indices do not have correct type");
    static_assert(sizeof...(_idx) == N, "Incorrect number of indices passed to operator()"); 
    return m_p_data[((_idx*m_strides[Is]) + ...)];
  }

  T* m_p_data; 

  std::array<std::size_t,N> m_dims;

  std::array<std::size_t,N> m_strides;

  std::size_t m_size;

};


template <int N, typename T>
class Block : public BlockSpan<N,T>
{
 public:
  
  Block(const std::array<std::size_t,N>& _block_sizes) 
    : BlockSpan<N,T>(nullptr,_block_sizes)
    , m_p_storage(nullptr)
  {
    m_p_storage.reset(new T[this->size()]);
    this->swap(m_p_storage.get());
  }

  Block(const Block& _block)
    : BlockSpan<N,T>(_block)
    , m_p_storage(nullptr)
  {
    m_p_storage.reset(new T[this->size()]);
    std::copy(_block.begin(), _block.end(), m_p_storage.get());
    this->swap(m_p_storage.get());
  }

  Block(Block&& _block) = default;

  Block& operator=(const Block& _block)
  {
    if (&_block == this) return *this;

    BlockSpan<N,T>::operator=(_block);
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