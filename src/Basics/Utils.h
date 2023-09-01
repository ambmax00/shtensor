#ifndef SHTENSOR_UTILS_H
#define SHTENSOR_UTILS_H

#include "Definitions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace Shtensor
{

namespace Utils 
{

template<typename>
struct ArraySize;

template<typename T, size_t N>
struct ArraySize<std::array<T,N>> 
{
  static constexpr size_t size = N;
};

// needed because some compilers do not like array.size() in templates
#define ARRAY_SSIZE(_array) \
  ArraySize<\
    typename std::remove_const<\
      typename std::remove_reference<\
        decltype(_array)\
      >::type\
    >::type\
  >::size

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
  Container strides = _sizes;
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
static constexpr inline int64_t roll_indices_impl(const StrideArray& _strides, 
                                                  Idx... _idx, std::index_sequence<Is...> const &)
{
  static_assert((is_valid_index_type<Idx>() && ...), 
                "Indices do not have correct type");

  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");

  static_assert(sizeof...(_idx) == ARRAY_SSIZE(_strides), 
                "Incorrect number of indices passed to function"); 

  return ((_idx*_strides[Is]) + ...);
}

template <class StrideArray, class... Idx>
static constexpr inline int64_t roll_indices(const StrideArray& _strides, 
                                             Idx... _idx)
{
  return roll_indices_impl<StrideArray,Idx...>(_strides, std::forward<Idx>(_idx)..., 
                                               std::make_index_sequence<ARRAY_SSIZE(_strides)>{}); 
}

template <class StrideArray, class IndexArray, std::size_t... Is>
static constexpr inline int64_t roll_index_array_impl(const StrideArray& _strides, 
                                                      const IndexArray& _index_array,
                                                      std::index_sequence<Is...> const&)
{
  static_assert(is_valid_index_type<typename IndexArray::value_type>(), 
                "Index array does not have correct size");

  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");
  
  static_assert(ARRAY_SSIZE(_strides) == ARRAY_SSIZE(_index_array),
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
  static_assert(ARRAY_SSIZE(_strides) == ARRAY_SSIZE(_indices), 
                "Stride array does not have same size as index array");

  static_assert(is_valid_index_type<typename IndexArray::value_type>(),
                "Index array does not have the correct type");
  
  static_assert(is_valid_index_type<typename StrideArray::value_type>(),
                "Stride array does not have the correct type");

  constexpr int64_t Dim = ARRAY_SSIZE(_strides);

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
static constexpr inline void unroll_index(const StrideArray& _strides, 
                                          LongIndexType _idx,
                                          IndexArray& _indices)
{
  unroll_index_impl(_strides, _idx, _indices, std::make_index_sequence<ARRAY_SSIZE(_strides)>{});
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
static constexpr inline auto varray_get_sizes(const MultiArray& _array)
{
  std::array<int64_t, ARRAY_SSIZE(_array)> out;
  std::generate(out.begin(), out.end(), 
    [&_array,i=0]() mutable { return Utils::ssize(_array[i++]); });
  return out;
}

template <class Integer>
static inline void divide_equally(Integer _nb_elements, int _bin, int _nb_bins, 
                                  Integer& _bin_start, Integer& _bin_end)
{
  // divide such that bin 0 takes the rest
  Integer rest = _nb_elements % _nb_bins;
  Integer bin_size_lower = _nb_elements / _nb_bins;
  Integer bin_offset = (_bin == 0) ? 0 : rest;

  Integer bin_size = (_bin == 0) ? bin_size_lower + rest : bin_size_lower; 
  _bin_start = _bin*bin_size_lower + bin_offset;
  _bin_end = _bin_start + bin_size;

  return;
}

template <class Array>
bool has_duplicates(const Array& _array)
{
  return (std::set(_array.begin(),_array.end()).size() != _array.size());
}

template <class Array1, class Array2>
bool contains_same_elements(const Array1& _array1, const Array2& _array2)
{
  Array1 _array1_copy = _array1;
  Array2 _array2_copy = _array2;

  std::sort(_array1_copy.begin(), _array1_copy.end());
  std::sort(_array2_copy.begin(), _array2_copy.end());

  return std::equal(_array1_copy.begin(), _array1_copy.end(), _array2_copy.begin());
}

template <class Array>
typename Array::value_type product(const Array& _array)
{
  return std::accumulate(_array.begin(),_array.end(),typename Array::value_type(1), 
                         std::multiplies<typename Array::value_type>{});
}


template <typename T, class SizeArray, class OrderArray>
void reshape(const T* _p_in, const SizeArray& _sizes, const OrderArray& _order, T* _p_out)
{
  const int dim = Utils::ssize(_sizes);
  const int64_t nb_elements = Utils::product(_sizes);

  // get reordered sizes
  const auto& ssizes = _sizes;
  auto rsizes = _sizes;
  
  for (int i = 0; i < Utils::ssize(_sizes); ++i)
  {
    rsizes[i] = ssizes[_order[i]];
  }

  const auto sstrides = compute_strides(ssizes);
  const auto rstrides = compute_strides(rsizes);

  std::vector<int> sindices(ssizes.size(),0);
  std::vector<int> rindices(ssizes.size(),0);

  for (int64_t sidx = 0; sidx < nb_elements; ++sidx)
  {
    int64_t sidx_tmp = sidx;
    for (int i = 0; i < dim; ++i)
    {
      sindices[dim-i-1] = sidx_tmp / sstrides[dim-i-1];
      sidx_tmp = sidx_tmp % sstrides[dim-i-1];
    }
    
    for (int i = 0; i < dim; ++i)
    {
      rindices[i] = sindices[_order[i]];
    }
    
    int64_t ridx = 0;
    for (int i = 0; i < dim; ++i)
    {
      ridx += rindices[i]*rstrides[i];
    }

    _p_out[ridx] = _p_in[sidx];
  }
  
}

template <class Vector>
Vector concat(const Vector& _v1, const Vector& _v2)
{
  Vector out = _v1;
  out.insert(out.end(),_v2.begin(),_v2.end());
  return out;
}

inline static std::vector<std::string> split(const std::string& _str, const std::string& _del)
{
  std::vector<std::string> out;

  std::size_t last = 0; 
  std::size_t next = 0; 
  while ((next = _str.find(_del, last)) != std::string::npos) 
  {   
    out.push_back(_str.substr(last, next-last));   
    last = next + 1; 
  } 
  
  out.push_back(_str.substr(last));

  return out;
}

template <class T>
typename std::enable_if<std::is_integral<T>::value,T>::type div_ceil(T _a, T _b)
{
  return static_cast<T>(std::ceil(static_cast<double>(_a)/static_cast<double>(_b)));
}

template <class T>
constexpr bool is_valid_float_type()
{
  if constexpr (std::is_same<T,float>::value || std::is_same<T,double>::value)
  {
    return true;
  }
  else 
  {
    return false;
  }
}

template <class T>
constexpr typename std::enable_if<is_valid_float_type<T>(),bool>::type bit_equal(T _a, T _b)
{
  const uint8_t* pa = reinterpret_cast<uint8_t*>(&_a);
  const uint8_t* pb = reinterpret_cast<uint8_t*>(&_b);

  return std::equal(pa, pa+sizeof(T), pb);
}

template <class T>
constexpr typename std::enable_if<std::is_integral<T>::value,T>::type 
round_next_multiple(T _val, T _factor)
{    
    T is_pos = (T)(_val >= 0);
    return ((_val + is_pos * (_factor - 1)) / _factor) * _factor;
}

} // end namespace Utils 

} // end namespace Shtensor


#endif