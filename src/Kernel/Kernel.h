#ifndef SHTENSOR_KERNEL_H
#define SHTENSOR_KERNEL_H

#include "KernelDefinitions.h"

#include <any>
#include <array>
#include <memory>
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

template <typename T>
static constexpr inline FloatType kernel_type()
{
  if constexpr (std::is_same<float,T>::value)
  {
    return FloatType::FLOAT32;
  }
  if constexpr (std::is_same<double,T>::value)
  {
    return FloatType::FLOAT64;
  }
}

class KernelImpl;

class KernelBase
{
 public:

  explicit KernelBase(const std::string _expr, 
                      const std::vector<int>& _sizes_in1, 
                      const std::vector<int>& _sizes_in2,
                      const std::vector<int>& _sizes_out,
                      AnyFloat _alpha, 
                      AnyFloat _beta,
                      FloatType _float_type,
                      KernelType _kernel_type);

  const AnyKernel& get_kernel_function();

  std::string get_info();

  ~KernelBase();

 protected: 

  std::unique_ptr<KernelImpl> mp_impl;

};

template <class T>
class Kernel : public KernelBase
{
 public: 

  using KernelFunctionT = std::function<int(T*,T*,T*,int64_t)>;

  template <class ArrayIn1, class ArrayIn2, class ArrayOut>
  explicit Kernel(const std::string _expr, 
                  const ArrayIn1& _sizes_in1, 
                  const ArrayIn2& _sizes_in2,
                  const ArrayOut& _sizes_out,
                  T _alpha, 
                  T _beta,
                  KernelType _kernel_type)
    : KernelBase(_expr, 
                 std::vector<int>(_sizes_in1.begin(), _sizes_in1.end()), 
                 std::vector<int>(_sizes_in2.begin(), _sizes_in2.end()), 
                 std::vector<int>(_sizes_out.begin(), _sizes_out.end()), 
                 _alpha, 
                 _beta, 
                 kernel_type<T>(),
                 _kernel_type)
  {
  }

  inline void call(T* _a, T* _b, T* _c, int64_t _nb_ops)
  {
    std::get<KernelFunctionT>(get_kernel_function())(_a,_b,_c,_nb_ops);
  }

};

} // end namespace Shtensor

#endif