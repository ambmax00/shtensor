#include "Kernel.h"

#include "asmjit/x86/x86assembler.h"
#include "LAPACK.h"

namespace Shtensor 
{

class KernelImpl
{
 public:

  explicit KernelImpl(const std::string _expr, 
                      const std::vector<int>& _sizes_in1, 
                      const std::vector<int>& _sizes_in2,
                      const std::vector<int>& _sizes_out,
                      std::any _alpha, 
                      std::any _beta,
                      KernelType _kernel_type,
                      KernelMethod _kernel_method)
    : m_expression(_expr)
    , m_sizes_in1(_sizes_in1)
    , m_sizes_in2(_sizes_in2)
    , m_sizes_out(_sizes_out)
    , m_alpha(_alpha)
    , m_beta(_beta)
    , m_type(_kernel_type)
    , m_method(_kernel_method)
    , m_info_in1()
    , m_info_in2()
    , m_info_out()
    , m_kernel_function()
  {
    
    std::string err_msg = "";

    if (!compute_contract_info(_expr, m_sizes_in1, m_sizes_in2, m_sizes_out, 
                               m_info_in1, m_info_in2, m_info_out, err_msg))
    {
      throw std::runtime_error(err_msg);
    }

    constexpr auto cenum = [](KernelType _type, KernelMethod _method)
    {
      return (static_cast<int>(_type) | static_cast<int>(_method));
    };

    int full_type = cenum(m_type, m_method);
    
    switch (full_type)
    {
      case cenum(KernelType::FLOAT32, KernelMethod::LAPACK):
      {
        m_kernel_function = create_kernel_lapack_float32();
        break;
      }
      case cenum(KernelType::FLOAT64, KernelMethod::LAPACK):
      {
        m_kernel_function = create_kernel_lapack_float64();
        break;
      }
      default:
      {
        throw std::runtime_error("Unknown kernel type");
      }
    }

  }

  KernelFunctionSp create_kernel_lapack_float32();

  KernelFunctionDp create_kernel_lapack_float64();

  std::any get_kernel_function() { return m_kernel_function; }

  std::string get_info();

 private: 

  std::string m_expression;

  std::vector<int> m_sizes_in1;

  std::vector<int> m_sizes_in2;

  std::vector<int> m_sizes_out;

  std::any m_alpha;

  std::any m_beta;

  KernelType m_type;

  KernelMethod m_method;

  ContractInfo m_info_in1;

  ContractInfo m_info_in2;

  ContractInfo m_info_out;

  std::any m_kernel_function;

  static inline thread_local std::vector<char> m_buffer;

};

KernelBase::KernelBase(const std::string _expr, 
                       const std::vector<int>& _sizes_in1, 
                       const std::vector<int>& _sizes_in2,
                       const std::vector<int>& _sizes_out,
                       std::any _alpha, 
                       std::any _beta,
                       KernelType _kernel_type,
                       KernelMethod _kernel_method)
  : mp_impl(std::make_unique<KernelImpl>(_expr, _sizes_in1, _sizes_in2, _sizes_out, _alpha, _beta,
            _kernel_type, _kernel_method))
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

KernelFunctionSp KernelImpl::create_kernel_lapack_float32()
{
  const int64_t size_a = Utils::product(m_sizes_in1);
  const int64_t size_b = Utils::product(m_sizes_in2);
  const int64_t size_c = Utils::product(m_sizes_out);

  const int64_t buffer_size = (size_a+size_b+size_c)*SSIZEOF(float);

  m_buffer.resize(buffer_size);

  float* p_buffer_a = reinterpret_cast<float*>(m_buffer.data());
  float* p_buffer_b = p_buffer_a + size_a;
  float* p_buffer_c = p_buffer_b + size_b;

  // THREAD SAFETY FOR MUTABLE BUFFER???
  auto func = [p_buffer_a,p_buffer_b,p_buffer_c,this](float* _a, float* _b, float* _c) -> int
  {
    const std::vector<int> order_a = Utils::concat(m_info_in1.map_row,m_info_in1.map_col);
    const std::vector<int> order_b = Utils::concat(m_info_in2.map_row,m_info_in2.map_col);
    const std::vector<int> order_c = Utils::concat(m_info_out.map_row,m_info_out.map_col);
  
    Utils::reshape(_a, m_sizes_in1, order_a, p_buffer_a);
    Utils::reshape(_b, m_sizes_in2, order_b, p_buffer_b);
    Utils::reshape(_c, m_sizes_out, order_c, p_buffer_c);

    const int m = Utils::ssize(m_info_in1.scatter_row);
    const int n = Utils::ssize(m_info_in2.scatter_col);
    const int k = Utils::ssize(m_info_in1.scatter_col);

    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = std::any_cast<float>(m_alpha);
    const float beta = std::any_cast<float>(m_beta);

    LAPACK::sgemm('N','N',m,n,k,alpha,p_buffer_a,lda,p_buffer_b,ldb,beta,
                  p_buffer_c,ldc);

    // reorder C
    std::vector<int> order_c_r(order_c.size());
    std::vector<int> sizes_c_r(order_c.size());

    for (int i = 0; i < Utils::ssize(order_c_r); ++i)
    {
      order_c_r[order_c[i]] = i;
      sizes_c_r[i] = m_sizes_out[order_c[i]];
    }

    Utils::reshape(p_buffer_c, sizes_c_r, order_c_r, _c);

    return 0;

  };

  return func;

}

KernelFunctionDp KernelImpl::create_kernel_lapack_float64()
{

  const int64_t size_a = Utils::product(m_sizes_in1);
  const int64_t size_b = Utils::product(m_sizes_in2);
  const int64_t size_c = Utils::product(m_sizes_out);

  const int64_t buffer_size = (size_a+size_b+size_c)*SSIZEOF(double);

  m_buffer.resize(buffer_size);

  double* p_buffer_a = reinterpret_cast<double*>(m_buffer.data());
  double* p_buffer_b = p_buffer_a + size_a;
  double* p_buffer_c = p_buffer_b + size_b;

  // THREAD SAFETY FOR MUTABLE BUFFER???
  auto func = [p_buffer_a,p_buffer_b,p_buffer_c,this]
              (double* _a, double* _b, double* _c) mutable -> int
  {
    const std::vector<int> order_a = Utils::concat(m_info_in1.map_row,m_info_in1.map_col);
    const std::vector<int> order_b = Utils::concat(m_info_in2.map_row,m_info_in2.map_col);
    const std::vector<int> order_c = Utils::concat(m_info_out.map_row,m_info_out.map_col);
  
    Utils::reshape(_a, m_sizes_in1, order_a, p_buffer_a);
    Utils::reshape(_b, m_sizes_in2, order_b, p_buffer_b);
    Utils::reshape(_c, m_sizes_out, order_c, p_buffer_c);

    const int m = Utils::ssize(m_info_in1.scatter_row);
    const int n = Utils::ssize(m_info_in2.scatter_col);
    const int k = Utils::ssize(m_info_in1.scatter_col);

    const int lda = m;
    const int ldb = k;
    const int ldc = n;

    const double alpha = std::any_cast<double>(m_alpha);
    const double beta = std::any_cast<double>(m_beta);

    LAPACK::dgemm('N','N',m,n,k,alpha,p_buffer_a,lda,p_buffer_b,ldb,beta,
                  p_buffer_c,ldc);

    // reorder C
    std::vector<int> order_c_r(order_c.size());
    std::vector<int> sizes_c_r(order_c.size());

    for (int i = 0; i < Utils::ssize(order_c_r); ++i)
    {
      order_c_r[order_c[i]] = i;
      sizes_c_r[i] = m_sizes_out[order_c[i]];
    }

    Utils::reshape(p_buffer_c, sizes_c_r, order_c_r, _c);

    return 0;

  };

  return func;

}

std::string KernelImpl::get_info()
{
  std::string out;

  out += fmt::format("Kernel Info\n");
  out += fmt::format("  Type: {}\n", static_cast<int>(m_type));
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