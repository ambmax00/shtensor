#ifndef SHTENSOR_KERNELIMPL
#define SHTENSOR_KERNELIMPL

#include "ContractInfo.h"

#include "asmjit/x86.h"
#include <any>

namespace Shtensor 
{

using KernelFunctionSp = std::function<int(float*,float*,float*,int64_t)>;

using KernelFunctionDp = std::function<int(double*,double*,double*,int64_t)>;

class KernelImpl
{
 public:

  explicit KernelImpl(const std::string _expr, 
                      const std::vector<int>& _sizes_in1, 
                      const std::vector<int>& _sizes_in2,
                      const std::vector<int>& _sizes_out,
                      std::any _alpha, 
                      std::any _beta,
                      FloatType _float_type,
                      KernelType _kernel_type);

  KernelFunctionSp create_kernel_lapack_float32();

  KernelFunctionDp create_kernel_lapack_float64();

  KernelFunctionSp create_kernel_xmm_float32();

  std::any get_kernel_function() { return m_kernel_function; }

  std::string get_info();

 private: 

  std::string m_expression;

  std::vector<int> m_sizes_in1;

  std::vector<int> m_sizes_in2;

  std::vector<int> m_sizes_out;

  std::any m_alpha;

  std::any m_beta;

  FloatType m_float_type;

  KernelType m_kernel_type;

  ContractInfo m_info_in1;

  ContractInfo m_info_in2;

  ContractInfo m_info_out;

  std::any m_kernel_function;

  std::vector<char> m_buffer;

  asmjit::JitRuntime m_jit_runtime;

  typedef int (*XmmFunc)(float*,float*,float*,int64_t);

  XmmFunc m_xmm_fn_holder;

  Log::Logger m_logger;
  
};

} //namespace Shtensor

#endif