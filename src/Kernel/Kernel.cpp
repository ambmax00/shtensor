#include "Kernel.h"
#include "BlockSpan.h"
#include "Utils.h"

#include "KernelImpl.h"

namespace Shtensor 
{

KernelBase::KernelBase(const std::string _expr, 
                       const std::vector<int>& _sizes_in1, 
                       const std::vector<int>& _sizes_in2,
                       const std::vector<int>& _sizes_out,
                       std::any _alpha, 
                       std::any _beta,
                       FloatType _float_type,
                       KernelType _kernel_method)
  : mp_impl(std::make_unique<KernelImpl>(_expr, _sizes_in1, _sizes_in2, _sizes_out, _alpha, _beta,
            _float_type, _kernel_method))
{

}

std::any KernelBase::get_kernel_function()
{
  return mp_impl->get_kernel_function();
}

std::string KernelBase::get_info()
{
  return mp_impl->get_info();
}

KernelBase::~KernelBase() = default;

} // end Shtensor