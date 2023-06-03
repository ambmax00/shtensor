#ifndef SHTENSOR_WINDOW_H
#define SHTENSOR_WINDOW_H

#include <functional>

namespace Shtensor
{

template <typename T>
class Window 
{
 public:

  using Deleter = std::function<void(T*)>;
  using Reallocator = std::function<T*(T*,int64_t)>;

  Window()
    : m_p_data(nullptr)
    , m_size(0)
    , m_deleter([]([[maybe_unused]]T* _ptr){ return; })
    , m_reallocator([](T* _ptr, [[maybe_unused]]int64_t _size) { return _ptr; })
  {
  }

  Window(T* _ptr, int64_t _size, Deleter _deleter, Reallocator _reallocator)
    : m_p_data(_ptr)
    , m_size(_size)
    , m_deleter(_deleter)
    , m_reallocator(_reallocator)
  {
  }

  Window(const Window& _win) = default;

  Window(Window&& _win) = default;

  Window& operator=(const Window& _win) = default;

  Window& operator=(Window&& _win) = default;

  ~Window() { if (m_deleter) m_deleter(m_p_data); }

  T& operator[](int64_t _index) { return m_p_data[_index]; }

  const T& operator[](int64_t _index) const { return m_p_data[_index]; }

  T* data() { return m_p_data; }

  const T* data() const { return m_p_data; }

  T* begin() { return m_p_data; }

  T* end() { return m_p_data + m_size; }

  const T* cbegin() const { return m_p_data; }

  const T* cend() const { return m_p_data + m_size; }

  void resize(int64_t _size) { m_p_data = m_reallocator(m_p_data, _size); }

 private:

  T* m_p_data;
  int64_t m_size;
  Deleter m_deleter;
  Reallocator m_reallocator;


};

} // end namespace Shtensor

#endif // SHTENSOR_WINDOW_H