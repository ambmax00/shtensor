#ifndef SHTENSOR_ALLOCATOR_H
#define SHTENSOR_ALLOCATOR_H

#include <functional>

namespace Shtensor
{

// A similar concept to std::allocator, but with relocate function
template <typename T>
class Allocator 
{
 public:

  using value_type = T;

  using Allocater = std::function<T*(int64_t)>;
  using Deleter = std::function<void(T*)>;
  using Reallocater = std::function<T*(T*,int64_t)>;

  Allocator(Allocater _f_alloc, Reallocater _f_realloc, Deleter _f_deleter)
    : m_f_alloc(_f_alloc)
    , m_f_realloc(_f_realloc)
    , m_f_deleter(_f_deleter) noexcept
  {
  }

  template <class U> 
  Allocator(const Allocator<U>&) noexcept;


  T* allocate (std::size_t n);
  void deallocate (T* p, std::size_t n);

 private:

  Allocater m_f_alloc;

  Reallocater m_f_realloc;

  Deleter m_f_deleter;


};

template <class T>
struct custom_allocator {
  using value_type = T;
  custom_allocator() noexcept;
  template <class U> custom_allocator (const custom_allocator<U>&) noexcept;
  T* allocate (std::size_t n);
  void deallocate (T* p, std::size_t n);
};

template <class T, class U>
constexpr bool operator== (const custom_allocator<T>&, const custom_allocator<U>&) noexcept;

template <class T, class U>
constexpr bool operator!= (const custom_allocator<T>&, const custom_allocator<U>&) noexcept;


} // namespace Shtensor

#endif // SHTENSOR_ALLOCATOR_H