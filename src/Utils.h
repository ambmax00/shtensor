#ifndef SHTENSOR_UTILS_H
#define SHTENSOR_UTILS_H

#include "Definitions.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <vector>

namespace Shtensor
{

namespace Utils 
{

template <class Container>
static inline constexpr int64_t ssize(const Container& _container)
{
  return static_cast<int64_t>(_container.size());
}

static inline std::vector<int> compute_default_dist (int nel, int nbin, std::vector<int> weights) 
{
  std::vector<int> dist_vec(weights.size(), 0);

  std::vector<int> occup(nbin, 0);
  int ibin = 0;

  for (int iel = 0; iel != nel; ++iel) 
  {
    int niter = 0;
    ibin = abs((ibin + 1) % nbin);

    while (occup[ibin] + weights[iel] >= *std::max_element(occup.begin(), occup.end())) 
    {
      int minloc = std::min_element(occup.begin(), occup.end()) - occup.begin();
      
      if (minloc == ibin) break;
      
      ibin = abs((ibin + 1) % nbin);
      
      ++niter;
    }

    dist_vec[iel] = ibin;
    occup[ibin] = occup[ibin] + weights[iel];
  }

  return dist_vec;
}

//template <int N, class Container>
//static inline std::array<Container::value_type,N> merge()

template <class Container>
static inline Container compute_strides(const Container& _sizes)
{
  Container strides;
  std::generate(strides.begin(), strides.end(), 
    [&_sizes,i=0]() mutable
    {
      typename Container::value_type stride = std::accumulate(_sizes.begin(), _sizes.begin()+i,
                                                typename Container::value_type(1), 
                                                std::multiplies<typename Container::value_type>());
      ++i;
      return stride;
    });
  return strides;
}

template <typename I>
static inline constexpr bool is_valid_index_type()
{
  return std::is_same<I,std::size_t>::value 
          || std::is_same<I,int>::value 
          || std::is_same<I,int64_t>::value;
}

template <class StrideArray, class... Idx, std::size_t... Is> 
static inline int64_t roll_indices_impl(const StrideArray& _strides, 
                                        Idx... _idx, std::index_sequence<Is...> const &)
{
  static_assert((is_valid_index_type<Idx>() && ...), 
                "Indices do not have correct type");

  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");

  static_assert(sizeof...(_idx) == _strides.size(), 
                "Incorrect number of indices passed to function"); 

  return ((_idx*_strides[Is]) + ...);
}

template <class StrideArray, class... Idx>
static inline int64_t roll_indices(const StrideArray& _strides, 
                                   Idx... _idx)
{
  return roll_indices_impl<StrideArray,Idx...>(_strides, std::forward<Idx>(_idx)..., 
                                               std::make_index_sequence<_strides.size()>{}); 
}

template <class StrideArray, class IndexArray, std::size_t... Is>
static inline int64_t roll_index_array_impl(const StrideArray& _strides, 
                                            const IndexArray& _index_array,
                                            std::index_sequence<Is...> const&)
{
  static_assert(is_valid_index_type<typename IndexArray::value_type>(), 
                "Index array does not have correct size");

  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");
  
  static_assert(_strides.size() == _index_array.size(),
                "Index array does not have same size as stride array");

  return ((int64_t(_index_array[Is])*int64_t(_strides[Is])) + ...);

}

template <class StrideArray, class IndexArray>
static inline int64_t roll_index_array(const StrideArray& _strides, 
                                       const IndexArray& _index_array)
{
  return roll_index_array_impl<StrideArray,IndexArray>(_strides,_index_array);
}

template <class StrideArray, class LongIndexType, class IndexArray, std::size_t... Is>
static inline void unroll_index_impl(const StrideArray& _strides, 
                                     LongIndexType _idx,
                                     IndexArray& _indices,
                                     std::index_sequence<Is...> const &)
{
  static_assert(_strides.size() == _indices.size(), 
                "Stride array does not have same size as index array");

  static_assert(is_valid_index_type<typename IndexArray::value_type>(),
                "Index array does not have the correct type");
  
  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");

  constexpr int64_t Dim = ssize(_strides);

  // We exploit the comma operator here to put everything in one single line
  // for the parameter pack expansion to do its magic
  // The expression is equivalent to:
  // for (i = 0; i < Dim; ++i)
  // {
  //   indices[Dim-i-1] = _idx / strides[Dim-i-1];
  //   _idx = _idx % strides[Dim-i-1];
  // }
  (((_indices[Dim-Is-1] = _idx / _strides[Dim-Is-1]), (_idx %= _strides[Dim-Is-1])) , ...);
}

template <class StrideArray, class LongIndexType, class IndexArray>
static inline void unroll_index(const StrideArray& _strides, 
                                LongIndexType _idx,
                                IndexArray& _indices)
{
  unroll_index_impl(_strides, _idx, _indices, std::make_index_sequence<_strides.size()>{});
}

template <int N>
using LoopIndexFunction = std::function<void(const std::array<int,N>& _loop_indices)>;

template <int N, int Step, class MultiArray>
static inline void loop_internal(const MultiArray& _array, const LoopIndexFunction<N>& _func, 
                                 std::array<int,N>& _idx)
{
  if constexpr (Step == N) 
  {
    _func(_idx);
  }
  else 
  {
    for (int64_t i = 0; i < Utils::ssize(_array[Step]); ++i)
    {
      _idx[Step] = i;
      loop_internal<N,Step+1,MultiArray>(_array, _func, _idx);
    }
  }
}

template <int N, class MultiArray>
static inline void loop_idx(const MultiArray& _array, const LoopIndexFunction<N>& _func)
{
  std::array<int,N> idx = {};
  loop_internal<N,0,MultiArray>(_array,_func,idx);
}

template <class MultiArray>
static inline int64_t varray_mult_ssize(const MultiArray& _array)
{
  return std::accumulate(_array.begin(), _array.end(), 1l, 
    [](int64_t sum, const auto& sub_array){ return sum*ssize(sub_array); });
}

template <class MultiArray>
static inline auto varray_get_sizes(const MultiArray& _array)
{
  std::array<int64_t,_array.size()> out;
  std::generate(out.begin(), out.end(), 
    [&_array,i=0]() mutable { return Utils::ssize(_array[i++]); });
  return out;
}

} // end namespace Utils 

} // end namespace Shtensor


#endif