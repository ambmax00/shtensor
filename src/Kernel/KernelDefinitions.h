#ifndef SHTENSOR_KERNEL_DEFINITIONS_H
#define SHTENSOR_KERNEL_DEFINITIONS_H

#include "Definitions.h"
#include <functional>
#include <variant>

namespace Shtensor
{

enum class KernelType
{
  LAPACK = 0,
  XMM = 1
};

enum AvxType
{
  AVX2 = 0
};

using KernelFunctionSp = std::function<int(float*,float*,float*,int64_t)>;

using KernelFunctionDp = std::function<int(double*,double*,double*,int64_t)>;

using AnyFloat = std::variant<float,double>;

using AnyKernel = std::variant<KernelFunctionSp,KernelFunctionDp>;

} // namespace Shtensor

#endif