#ifndef SHTENSOR_DEFINITIONS_H
#define SHTENSOR_DEFINITIONS_H

#include <cstdint>
#include <limits>

namespace Shtensor 
{

using uint8_t = std::uint8_t;
using size_t = std::size_t;

using int8_t = std::int8_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;

template <size_t usize> 
static inline constexpr int64_t ssizeof_impl() 
{
  static_assert(usize <= std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(usize);
}

#define SSIZEOF(type_or_expr) (ssizeof_impl<sizeof(type_or_expr)>())

} // end namespace Shtensor

#endif
