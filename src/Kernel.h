#ifndef SHTENSOR_KERNEL_H
#define SHTENSOR_KERNEL_H

#include "ContractInfo.h"

#include <array>
#include <string>
#include <vector>

namespace Shtensor
{

// Computes the tensor contraction 
// e.g. C(mj) = A(ijk) B(kim)
// It converts it to a matrix product 
// C(I,J) = A(I,K) B(K,J)
// Where I,J,K are mapped indices

enum KernelType
{
  KERNEL_SCATTER = 0
};

template <typename T>
class Kernel 
{
 public: 

  template <class ArrayIn1, class ArrayIn2, class ArrayOut>
  Kernel(const std::string _expr, 
         const ArrayIn1& _sizes_in1, 
         const ArrayIn2& _sizes_in2,
         const ArrayOut& _sizes_out,
         T _alpha, 
         T _beta,
         KernelType _kernel_type = KERNEL_SCATTER)
    : m_sizes_in1(_sizes_in1.begin(),_sizes_in1.end())
    , m_sizes_in2(_sizes_in2.begin(),_sizes_in2.end())
    , m_sizes_out(_sizes_out.begin(),_sizes_out.end())
    , m_alpha(_alpha)
    , m_beta(_beta)
    , m_logger(Log::create("Kernel"))
  {
    std::string err_msg;
    bool valid = compute_contract_info(_expr,m_sizes_in1,m_sizes_in2,m_sizes_out,m_info_in1,
                                       m_info_in2,m_info_out,err_msg);

    m_info_in1.print(m_logger);
    m_info_in2.print(m_logger);
    m_info_out.print(m_logger);

    if (!valid)
    {
      Log::print(m_logger, "Contract info failed: {}\n", err_msg);
    }
    

  }

 private:

  // using vector so I do not have to carry around three integers as template parameters...
  std::vector<int> m_sizes_in1;
  std::vector<int> m_sizes_in2;
  std::vector<int> m_sizes_out;

  T m_alpha;
  T m_beta;

  ContractInfo m_info_in1;
  ContractInfo m_info_in2;
  ContractInfo m_info_out;

  Log::Logger m_logger;

};

} // end namespace Shtensor

#endif