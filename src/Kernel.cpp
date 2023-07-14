#include "Kernel.h"

#include "asmjit/x86/x86assembler.h"
#include "LAPACK.h"

namespace Shtensor 
{

KernelFunctionSp KernelImpl::create_kernel_lapack_float()
{

  // THREAD SAFETY FOR MUTABLE BUFFER???
  auto func = [buffer_a=std::vector<float>(Utils::product(m_sizes_in1)),
               buffer_b=std::vector<float>(Utils::product(m_sizes_in2)),
               buffer_c=std::vector<float>(Utils::product(m_sizes_out)),
                this]
              (float* _a, float* _b, float* _c) mutable -> int
  {
    const std::vector<int> order_a = Utils::concat(m_info_in1.map_row,m_info_in1.map_col);
    const std::vector<int> order_b = Utils::concat(m_info_in2.map_row,m_info_in2.map_col);
    const std::vector<int> order_c = Utils::concat(m_info_out.map_row,m_info_out.map_col);
  
    Utils::reshape(_a, m_sizes_in1, order_a, buffer_a.data());
    Utils::reshape(_b, m_sizes_in2, order_b, buffer_b.data());
    Utils::reshape(_c, m_sizes_out, order_c, buffer_c.data());

    const int m = Utils::ssize(m_info_in1.scatter_row);
    const int n = Utils::ssize(m_info_in2.scatter_col);
    const int k = Utils::ssize(m_info_in1.scatter_col);

    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = std::any_cast<float>(m_alpha);
    const float beta = std::any_cast<float>(m_beta);

    LAPACK::sgemm('N','N',m,n,k,alpha,buffer_a.data(),lda,buffer_b.data(),ldb,beta,
                  buffer_c.data(),ldc);

    // reorder C
    std::vector<int> order_c_r(order_c.size());
    std::vector<int> sizes_c_r(order_c.size());

    for (int i = 0; i < Utils::ssize(order_c_r); ++i)
    {
      order_c_r[order_c[i]] = i;
      sizes_c_r[i] = m_sizes_out[order_c[i]];
    }

    Utils::reshape(buffer_c.data(), sizes_c_r, order_c_r, _c);

    return 0;

  };

  return func;

}

KernelFunctionDp KernelImpl::create_kernel_lapack_double()
{

  // THREAD SAFETY FOR MUTABLE BUFFER???
  auto func = [buffer_a=std::vector<double>(Utils::product(m_sizes_in1)),
               buffer_b=std::vector<double>(Utils::product(m_sizes_in2)),
               buffer_c=std::vector<double>(Utils::product(m_sizes_out)),
                this]
              (double* _a, double* _b, double* _c) mutable -> int
  {
    const std::vector<int> order_a = Utils::concat(m_info_in1.map_row,m_info_in1.map_col);
    const std::vector<int> order_b = Utils::concat(m_info_in2.map_row,m_info_in2.map_col);
    const std::vector<int> order_c = Utils::concat(m_info_out.map_row,m_info_out.map_col);
  
    Utils::reshape(_a, m_sizes_in1, order_a, buffer_a.data());
    Utils::reshape(_b, m_sizes_in2, order_b, buffer_b.data());
    Utils::reshape(_c, m_sizes_out, order_c, buffer_c.data());

    const int m = Utils::ssize(m_info_in1.scatter_row);
    const int n = Utils::ssize(m_info_in2.scatter_col);
    const int k = Utils::ssize(m_info_in1.scatter_col);

    const int lda = m;
    const int ldb = k;
    const int ldc = n;

    const double alpha = std::any_cast<double>(m_alpha);
    const double beta = std::any_cast<double>(m_beta);

    LAPACK::dgemm('N','N',m,n,k,alpha,buffer_a.data(),lda,buffer_b.data(),ldb,beta,
                  buffer_c.data(),ldc);

    // reorder C
    std::vector<int> order_c_r(order_c.size());
    std::vector<int> sizes_c_r(order_c.size());

    for (int i = 0; i < Utils::ssize(order_c_r); ++i)
    {
      order_c_r[order_c[i]] = i;
      sizes_c_r[i] = m_sizes_out[order_c[i]];
    }

    Utils::reshape(buffer_c.data(), sizes_c_r, order_c_r, _c);

    return 0;

  };

  return func;

}

std::string KernelImpl::get_info()
{
  std::string out;

  out += fmt::format("Kernel Info\n");
  out += fmt::format("  Type: {}\n", m_type_info.name());
  out += fmt::format("  Expression: {}\n", m_expression);

  out += fmt::format("Info Tensor In 1\n");
  out += fmt::format("  Sizes: {}\n", fmt::join(m_sizes_in1.begin(), m_sizes_in1.end(), ","));
  out += fmt::format(m_info_in1.get_info());

  out += fmt::format("Info Tensor In 2\n");
  out += fmt::format("  Sizes: {}\n", fmt::join(m_sizes_in2.begin(), m_sizes_in2.end(), ","));
  out += fmt::format(m_info_in2.get_info());

  out += fmt::format("Info Tensor Out\n");
  out += fmt::format("  Sizes: {}\n", fmt::join(m_sizes_out.begin(), m_sizes_out.end(), ","));
  out += fmt::format(m_info_out.get_info());

  return out;
}

} // end Shtensor