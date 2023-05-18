#ifndef SHTENSOR_WINDOW_H
#define SHTENSOR_WINDOW_H

#include <memory>

namespace Shtensor
{

template <typename T>
class Window 
{
 public:

  Window()
    : m_p_data(nullptr)
    , m_size(0)
  {
  }

  Window(const std::shared_ptr<T>& _ptr, int64_t _size)
    : m_p_data(_ptr)
    , m_size(_size)
  {
  }

  Window(const Window& _win) = default;

  Window(Window&& _win) = default;

  Window& operator=(const Window& _win) = default;

  Window& operator=(Window&& _win) = default;

  ~Window() {}

  T& operator[](int64_t _index) { return m_p_data.get()[_index]; }

  const T& operator[](int64_t _index) const { return m_p_data.get()[_index]; }

  T* data() { return m_p_data; }

  const T* data() const { return m_p_data.get(); }

  T* begin() { return m_p_data.get(); }

  T* end() { return m_p_data.get() + m_size; }

  const T* cbegin() const { return m_p_data.get(); }

  const T* cend() const { return m_p_data.get() + m_size; }

 private:

  std::shared_ptr<T> m_p_data;
  int64_t m_size;

};

} // end namespace Shtensor

#endif // SHTENSOR_WINDOW_H