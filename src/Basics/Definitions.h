#ifndef SHTENSOR_DEFINITIONS_H
#define SHTENSOR_DEFINITIONS_H

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>

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

template <size_t usize>
static inline constexpr int isizeof_impl()
{
  static_assert(usize <= std::numeric_limits<int>::max());
  return static_cast<int>(usize);
}

#define SSIZEOF(type_or_expr) (Shtensor::ssizeof_impl<sizeof(type_or_expr)>())

#define ISIZEOF(type_or_expr) (Shtensor::isizeof_impl<sizeof(type_or_expr)>())

template <typename T>
static inline constexpr MPI_Datatype get_mpi_type()
{
  #define EXPR(ctype, mtype) \
    if (std::is_same<T,ctype>::value)\
    {\
      return mtype;\
    }

  EXPR(char, MPI_CHAR)
  EXPR(unsigned char, MPI_UNSIGNED_CHAR)
  EXPR(wchar_t, MPI_WCHAR)
  EXPR(short, MPI_SHORT)
  EXPR(unsigned short, MPI_UNSIGNED_SHORT)
  EXPR(int, MPI_INT)
  EXPR(unsigned, MPI_UNSIGNED)
  EXPR(long, MPI_LONG)
  EXPR(unsigned long, MPI_UNSIGNED_LONG)
  EXPR(long long, MPI_LONG_LONG)
  EXPR(unsigned long long, MPI_UNSIGNED_LONG_LONG)
  EXPR(float, MPI_FLOAT)
  EXPR(double, MPI_DOUBLE)
  EXPR(long double, MPI_LONG_DOUBLE)
  EXPR(int8_t, MPI_INT8_T)
  EXPR(int16_t, MPI_INT16_T)
  EXPR(int32_t, MPI_INT32_T)
  EXPR(int64_t, MPI_INT64_T)
  EXPR(uint8_t, MPI_UINT8_T)
  EXPR(uint16_t, MPI_UINT16_T)
  EXPR(uint32_t, MPI_UINT32_T)
  EXPR(uint64_t, MPI_UINT64_T)
  EXPR(bool, MPI_C_BOOL)

  static_assert(MPI_DATATYPE_NULL, "Unknown MPI datatype");

  return MPI_DATATYPE_NULL;

  #undef EXPR
}

constexpr static inline int64_t KiB = 1024;
constexpr static inline int64_t MiB = KiB*1024;
constexpr static inline int64_t GiB = MiB*1024;
constexpr static inline int64_t TiB = GiB*1024; // will probably never be used, but one can dream

enum class FloatType 
{
  FLOAT32 = 0,
  FLOAT64 = 1
};

template <int N>
using VArray = std::array<std::vector<int>,N>;

template <typename T>
using VVector = std::vector<std::vector<T>>;

template <typename T>
using VPVector = std::vector<std::vector<T>*>;

} // end namespace Shtensor

#endif
