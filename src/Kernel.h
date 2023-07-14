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

enum KernelType
{
  KERNEL_LAPACK = 0
};

class KernelImpl
{
 public:

  template <class T, class ArrayIn1, class ArrayIn2, class ArrayOut>
  explicit KernelImpl(const std::string _expr,
                      const ArrayIn1& _sizes_in1, 
                      const ArrayIn2& _sizes_in2,
                      const ArrayOut& _sizes_out,
                      T _alpha, 
                      T _beta,
                      KernelType _kernel_type)
    : m_expression(_expr)
    , m_sizes_in1(_sizes_in1.begin(),_sizes_in1.end())
    , m_sizes_in2(_sizes_in2.begin(),_sizes_in2.end())
    , m_sizes_out(_sizes_out.begin(),_sizes_out.end())
    , m_alpha(_alpha)
    , m_beta(_beta)
    , m_kernel_type(_kernel_type)
    , m_type_info(typeid(T))
    , m_logger(Log::create("KernelImpl"))
  {
    static_assert(std::is_same<T,double>::value || std::is_same<T,float>::value, 
                  "Only float and double allowed.");

    std::string err_msg;
    bool valid = compute_contract_info(_expr,m_sizes_in1,m_sizes_in2,m_sizes_out,m_info_in1,
                                       m_info_in2,m_info_out,err_msg);

    if (!valid)
    {
      throw fmt::format("Contract info failed: {}\n", err_msg);
    }
  }

  KernelFunctionSp create_kernel_lapack_float();

  KernelFunctionDp create_kernel_lapack_double();

  std::string get_info();

 protected:

  const std::string m_expression; 

  // using vector so I do not have to carry around three integers as template parameters...
  std::vector<int> m_sizes_in1;
  std::vector<int> m_sizes_in2;
  std::vector<int> m_sizes_out;

  std::any m_alpha;
  std::any m_beta;

  ContractInfo m_info_in1;
  ContractInfo m_info_in2;
  ContractInfo m_info_out;

  KernelType m_kernel_type;

  const std::type_info& m_type_info;

  Log::Logger m_logger;

};

template <class T>
class Kernel : public KernelImpl
{
 public: 

  template <class ArrayIn1, class ArrayIn2, class ArrayOut>
  explicit Kernel(const std::string _expr, 
                  const ArrayIn1& _sizes_in1, 
                  const ArrayIn2& _sizes_in2,
                  const ArrayOut& _sizes_out,
                  T _alpha, 
                  T _beta,
                  KernelType _kernel_type = KERNEL_LAPACK)
    : KernelImpl(_expr, _sizes_in1, _sizes_in2, _sizes_out, _alpha, _beta, _kernel_type) 
  {
  }

};

} // end namespace Shtensor

#endif