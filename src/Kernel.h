#ifndef SHTENSOR_KERNEL_H
#define SHTENSOR_KERNEL_H

#include "ContractInfo.h"

#include <any>
#include <array>
#include <string>
#include <typeinfo>
#include <vector>

namespace Shtensor
{

// Computes the tensor contraction 
// e.g. C(mj) = A(ijk) B(kim)
// It converts it to a matrix product 
// C(I,J) = A(I,K) B(K,J)
// Where I,J,K are mapped indices

using KernelFunctionSp = std::function<int(float*,float*,float*)>;

using KernelFunctionDp = std::function<int(double*,double*,double*)>;

enum class KernelMethod
{
  INVALID = 0x000000,
  LAPACK = 0x000001,
  XMM = 0x000002
};

enum class KernelType
{
  INVALID = 0x000000,
  FLOAT32 = 0x000100,
  FLOAT64 = 0x000200
};

template <typename T>
static constexpr inline KernelType kernel_type()
{
  if constexpr (std::is_same<float,T>::value)
  {
    return KernelType::FLOAT32;
  }
  if constexpr (std::is_same<double,T>::value)
  {
    return KernelType::FLOAT64;
  }
  return KernelType::INVALID; 
}

class KernelImpl;

class KernelBase
{
 public:

  explicit KernelBase(const std::string _expr, 
                      const std::vector<int>& _sizes_in1, 
                      const std::vector<int>& _sizes_in2,
                      const std::vector<int>& _sizes_out,
                      std::any _alpha, 
                      std::any _beta,
                      KernelType _kernel_type,
                      KernelMethod _kernel_method);

  std::any get_kernel_function();

  std::string get_info();

  ~KernelBase();

 protected: 

  std::unique_ptr<KernelImpl> mp_impl;

};

template <class T>
class Kernel : public KernelBase
{
 public: 

  using KernelFunctionT = std::function<int(T*,T*,T*)>;

  template <class ArrayIn1, class ArrayIn2, class ArrayOut>
  explicit Kernel(const std::string _expr, 
                  const ArrayIn1& _sizes_in1, 
                  const ArrayIn2& _sizes_in2,
                  const ArrayOut& _sizes_out,
                  T _alpha, 
                  T _beta,
                  KernelMethod _kernel_method)
    : KernelBase(_expr, 
                 std::vector<int>(_sizes_in1.begin(), _sizes_in1.end()), 
                 std::vector<int>(_sizes_in2.begin(), _sizes_in2.end()), 
                 std::vector<int>(_sizes_out.begin(), _sizes_out.end()), 
                 _alpha, 
                 _beta, 
                 kernel_type<T>(),
                 _kernel_method)
    , m_kernel_function(std::any_cast<KernelFunctionT>(get_kernel_function()))
  {
  }

  inline void call(T* _a, T* _b, T* _c)
  {
    fmt::print("A: {:p} B: {:p} C_ {:p}\n", (void*)_a, (void*)_b, (void*)_c);
    int res = m_kernel_function(_a,_b,_c);
    fmt::print("Result is {}\n", res);
  }

 private:

  KernelFunctionT m_kernel_function;

};

} // end namespace Shtensor

#endif