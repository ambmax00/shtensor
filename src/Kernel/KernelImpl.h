#ifndef SHTENSOR_KERNELIMPL
#define SHTENSOR_KERNELIMPL

#include "ContractInfo.h"
#include "KernelDefinitions.h"

#include "asmjit/x86.h"
#include <any>

namespace Shtensor 
{

class KernelImpl
{
 public:

  explicit KernelImpl(const std::string _expr, 
                      const std::vector<int>& _sizes_in1, 
                      const std::vector<int>& _sizes_in2,
                      const std::vector<int>& _sizes_out,
                      AnyFloat _alpha, 
                      AnyFloat _beta,
                      FloatType _float_type,
                      KernelType _kernel_type);

  void create_kernel_lapack_float32();

  void create_kernel_lapack_float64();

  void create_kernel_xmm_float32();

  const AnyKernel& get_kernel_function() { return m_kernel_function; }

  std::string get_info();

 private: 

  std::string m_expression;

  std::vector<int> m_sizes_in1;

  std::vector<int> m_sizes_in2;

  std::vector<int> m_sizes_out;

  AnyFloat m_alpha;

  AnyFloat m_beta;

  FloatType m_float_type;

  KernelType m_kernel_type;

  ContractInfo m_info_in1;

  ContractInfo m_info_in2;

  ContractInfo m_info_out;

  AnyKernel m_kernel_function;

  std::vector<char> m_buffer;

  asmjit::JitRuntime m_jit_runtime;

  typedef int (*XmmFuncFloat32)(float*,float*,float*,int64_t);

  typedef int (*XmmFuncFloat64)(double*,double*,double*,int64_t);

  void* m_p_function;

  Log::Logger m_logger;
  
};

} //namespace Shtensor

#endif