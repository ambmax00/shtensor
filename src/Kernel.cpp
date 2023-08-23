#include "Kernel.h"
#include "BlockSpan.h"

#include "asmjit/x86.h"
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
    , m_jit_runtime()
    , m_logger(Log::create("KernelImpl"))
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
      case cenum(KernelType::FLOAT32, KernelMethod::XMM):
      {
        m_kernel_function = create_kernel_xmm_float32();
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

  KernelType m_type;

  KernelMethod m_method;

  ContractInfo m_info_in1;

  ContractInfo m_info_in2;

  ContractInfo m_info_out;

  std::any m_kernel_function;

  std::vector<char> m_buffer;

  asmjit::JitRuntime m_jit_runtime;

  Log::Logger m_logger;
  
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

#if 0
KernelFunctionSp KernelImpl::create_kernel_xmm_float32_old()
{
  using namespace asmjit;
  using namespace x86;

  CodeHolder code; 
  code.init(m_jit_runtime.environment(), m_jit_runtime.cpuFeatures());

  x86::Assembler assembler(&code);

  /*
   *       c    +=    a a a a  * b
   *       c          a a a a    b
   *       c          a a a a    b
   *       c          a a a a    b
   */

  // Get number of registers along I
  const int m = Utils::ssize(m_info_in1.scatter_row);
  const int n = Utils::ssize(m_info_in2.scatter_col);
  const int k = Utils::ssize(m_info_in1.scatter_col);

  const float alpha = std::any_cast<float>(m_alpha);
  const float beta = std::any_cast<float>(m_beta);

  if (!m_jit_runtime.cpuFeatures().x86().hasAVX2())
  {
    throw std::runtime_error("CPU does not have AVX2 capabilities");
  }

  // ymm register size
  const int regSizeByte = 32;

  // how many units in a register?
  const int nb_floats_reg = regSizeByte/sizeof(float);
  
  // how many registers in one chunk at maximum?
  const int nb_max_reg_vchunk_m = 1;
  const int nb_max_reg_hchunk_k = 4;

  // actual number of registers in chunk
  const int nb_reg_vchunk_m = std::min(Utils::div_ceil(m, nb_floats_reg), 
                                       nb_max_reg_vchunk_m);

  const int nb_reg_hchunk_k = std::min(k,nb_max_reg_hchunk_k);

  // number of chunks
  const int nb_vchunks_m = Utils::div_ceil(m, nb_reg_vchunk_m*nb_floats_reg);
  const int nb_hchunks_k = Utils::div_ceil(k, nb_reg_hchunk_k);

  // rest for creating mask
  const int mask_size_zero = nb_reg_vchunk_m*nb_vchunks_m*nb_floats_reg - m;

  DEBUG_VAR(m_logger, nb_reg_vchunk_m);
  DEBUG_VAR(m_logger, nb_reg_hchunk_k);
  DEBUG_VAR(m_logger, nb_vchunks_m);
  DEBUG_VAR(m_logger, nb_hchunks_k);
  DEBUG_VAR(m_logger, mask_size_zero);
  Log::debug(m_logger, "Bit mask: {:b}", bit_mask);

  // RDI: address of A
  // RSI: address of B
  // RDX: address of C
  int stack_offset = 0;

  // allocate space for stack variables

  // address to A
  const int stack_a = (stack_offset -= 8);
  // address to B
  const int stack_b = (stack_offset -= 8);
  // address to C
  const int stack_c = (stack_offset -= 8);

  // N loop variable
  const int stack_jloop = (stack_offset -= 4);
  // M loop variable
  const int stack_iloop = (stack_offset -= 4);
  // K loop variable
  const int stack_kloop = (stack_offset -= 4);

  // alpha and beta
  const int stack_alpha = (stack_offset -= 4);
  const int stack_beta = (stack_offset -= 4);

  // // Current position in (scatter) matrix A
  // const int stack_a_pos = (stack_offset -= 4);
  // // Current position in (scatter) matrix B
  // const int stack_b_pos = (stack_offset -= 4);
  // // CUrrent position in (scatter) matrix C
  // const int stack_c_pos = (stack_offset -= 4);

  const int stack_size = Utils::round_next_multiple(-stack_offset, 16);

  Log::debug(m_logger, "Stack: {}\n", stack_size);

  // organize registers

  int regID = 16;
  auto fetch_reg = [&regID]()
  {
    regID--;
    if (regID < 0)
    {
      throw std::runtime_error("No ymm registers available");
    }
    return Ymm(regID);
  };

  // registers to hold block CIj
  std::vector<Ymm> ymm_c(nb_reg_vchunk_m);
  std::generate(ymm_c.begin(), ymm_c.end(), fetch_reg);

  // registers to hold block AIK
  std::vector<Ymm> regs_a(nb_reg_vchunk_m*nb_reg_hchunk_k);
  std::generate(regs_a.begin(), regs_a.end(), fetch_reg);

  // register for alpha
  Ymm ymm_alpha = fetch_reg();

  // register for beta
  Ymm ymm_beta = fetch_reg();

  // mask 
  Ymm ymm_mask = fetch_reg();

  // tmp 
  Ymm ymm_tmp = fetch_reg();

  // ======== PRECOMPUTE OFFSETS IN MEMORY ===========

  m_buffer.resize(((m*k)+(k*n)+(m*n)+2*nb_floats_reg)*sizeof(int));
  auto scatter_idx_a = Span<int>{reinterpret_cast<int*>(m_buffer.data()), m*k};
  auto scatter_idx_b = Span<int>{scatter_idx_a.data() + m*k, k*n};
  auto scatter_idx_c = Span<int>{scatter_idx_b.data() + k*n, m*n};
  auto nomask_values = Span<int>(scatter_idx_c.data(), nb_floats_reg);
  auto mask_values = Span<int>(nomask_values.data(), nb_floats_reg);

  auto compute_scatter_indices = 
  [](auto& _pdest, const auto& _scatter_row, const auto& _scatter_col)
  {
    const int nb_row = Utils::ssize(_scatter_row);
    const int nb_col = Utils::ssize(_scatter_col);

    for (int j = 0; j < nb_col; ++j)
    {
      for (int i = 0; i < nb_row; ++i)
      {
        const int val = _scatter_row[i] + _scatter_col[j];
        if (val >= nb_row*nb_col)
        {
          throw std::runtime_error("Scatter index overflow");
        }
        _pdest[i+j*nb_row] = val;
      }
    }
  };

  compute_scatter_indices(scatter_idx_a, m_info_in1.scatter_row, m_info_in1.scatter_col);
  compute_scatter_indices(scatter_idx_b, m_info_in2.scatter_row, m_info_in2.scatter_col);
  compute_scatter_indices(scatter_idx_c, m_info_out.scatter_row, m_info_out.scatter_col);

  Log::debug(m_logger, "Scatter idx C: {}", 
    fmt::join(scatter_idx_c.begin(), scatter_idx_c.end(), ","));

  // Store mask for vector loading
  std::fill(nomask_values.begin(), )

  std::fill(mask_values.begin(), mask_values.end(), 0);
  for (int f = 0; f < nb_floats_reg-mask_size_zero; ++f)
  {
    mask_values[f] = 0xFFFFFFFF;
  }

  // ======== START ASSEMBLING FUNCTION =============

  // put stack pointer into rbp
  assembler.push(regs::rbp);
  assembler.mov(regs::rbp, regs::rsp);

  //assembler.sub(regs::rsp, Imm{stack_size});

  // static const char* formatString = "Hello\n";

  // assembler.mov(regs::eax, Imm{0});
  // assembler.mov(regs::rdi, Imm{(void*)formatString});

  // assembler.call(printf);

  // put addresses of arrays into stack
  assembler.mov(qword_ptr(regs::rbp, stack_a), regs::rdi);
  assembler.mov(qword_ptr(regs::rbp, stack_b), regs::rsi);
  assembler.mov(qword_ptr(regs::rbp, stack_c), regs::rdx);

  // put alpha and beta into stack
  Log::debug(m_logger, "alpha={}, beta={}", alpha, beta);

  assembler.mov(dword_ptr(regs::rbp, stack_alpha), Imm{alpha});
  assembler.mov(dword_ptr(regs::rbp, stack_beta), Imm{beta});

  // load alpha and beta
  assembler.vbroadcastss(ymm_alpha, Mem{regs::rbp, stack_alpha});
  assembler.vbroadcastss(ymm_beta, Mem{regs::rbp, stack_beta});

  // addresses for offsets
  const int64_t scatter_idx_begin_a = reinterpret_cast<int64_t>(scatter_idx_a.data());
  const int64_t scatter_idx_begin_b = reinterpret_cast<int64_t>(scatter_idx_b.data());
  const int64_t scatter_idx_begin_c = reinterpret_cast<int64_t>(scatter_idx_c.data());

  // C kernel
    // Loop over n: i = 0 ---> n
      // Loop over m: j = 0 ---> m/vecsize * vecsize
        // Load C(I,j)
        // Call K kernel
        // Store C(I,j)
      // end
      // Epilogue loop over m: j (rest)
        // Maskload C(I,j)
        // Call mask K kernel
        // Maskstore C(I,j)
      // end epilogue
    // end loop n
  // end KERNEL

  Log::debug(m_logger, "Address for scatter c: {}", (void*)scatter_idx_c.data());

  // ======= LOOP OVER COLUMNS J ==============  

  // init loop var
  assembler.mov(dword_ptr(regs::rbp, stack_jloop), 0);

  Label jloop_label = assembler.newLabel();
  assembler.bind(jloop_label);

  { // ======= LOOP OVER CHUNKS M =========
    
    // init loop var
    assembler.mov(dword_ptr(regs::rbp, stack_iloop), 0);

    Label iloop_label = assembler.newLabel();
    assembler.bind(iloop_label);

    // Load chunk C(MREG,j)
    for (int ireg = 0; ireg < nb_reg_vchunk_m; ++ireg)
    {
      if (!Utils::bit_equal(beta,0.0f))
      {
        // check if last chunk
        assembler.mov()
        assembler.cmp(dword_ptr(regs::rbp, stack_iloop), m-ireg*nb_floats_reg);
        assembler.seta(regs::al);
        
        // move bit mask to xmm
        assembler.vmovd(ymm_mask.half(), regs::eax);

        // i + j*m => eax
        assembler.mov(regs::eax, Mem{regs::rbp, stack_jloop});
        assembler.imul(regs::eax, m);
        assembler.mov(regs::edx, regs::eax);
        assembler.mov(regs::eax, Mem{regs::rbp, stack_iloop});
        assembler.add(regs::eax, regs::edx);
        assembler.cdqe();

        assembler.mov(regs::rdx, (void*)scatter_idx_c.data());

        // get address to scatter index
        const uint32_t offset = ireg*nb_floats_reg*sizeof(int);
        assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, offset});

        // base address to c scatter idx
        assembler.mov(regs::rdx, (void*)scatter_idx_c.data());

        // move scatter indices into ymm
        assembler.vmovdqu(ymm_tmp, Mem{regs::rax});



        // load C register using scatter indices
        assembler.vgatherdps(ymm_c[ireg], Mem{regs::rdx, regs::rax, 0, 0});

        assembler.vmulps(ymm_c[ireg], ymm_c[ireg], ymm_beta);
      }
      else 
      {
        // beta is zero, so just zero out the registers
        assembler.vpxor(ymm_c[ireg], ymm_c[ireg], ymm_c[ireg]);
      }

    }


    // {
    //   // ======== LOOP OVER CHUNKS K =========

    //   // init loop var
    //   assembler.mov(dword_ptr(regs::rbp, stack_kloop), 0);

    //   // Load kchunk registers a(MREG,k), Load kchunk registers b(k,j)
    //   Label kregloop_label = assembler.newLabel();
    //   assembler.bind(kregloop_label);
      
    //   assembler.add(dword_ptr(regs::rbp, stack_kloop), Imm{nb_reg_hchunk_k});
    //   assembler.cmp(dword_ptr(regs::rbp, stack_kloop), k);
    //   assembler.jl(kregloop_label);

    // } // =============== end loop over K chunks ======================

    // store chunk C(MREG,j)
    for (int ireg = 0; ireg < nb_reg_vchunk_m; ++ireg)
    {
      // i + j*m => eax
      assembler.mov(regs::eax, Mem{regs::rbp, stack_jloop});
      assembler.imul(regs::eax, m);
      assembler.mov(regs::edx, regs::eax);
      assembler.mov(regs::eax, Mem{regs::rbp, stack_iloop});
      assembler.add(regs::eax, regs::edx);
      assembler.cdqe();

      assembler.mov(regs::rdx, (void*)scatter_idx_c.data());

      // get address to scatter index
      const uint32_t offset = ireg*nb_floats_reg*sizeof(int);
      assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, offset});

      // Store elements in ymm to C
      for (int f = 0; f < std::min(nb_reg_hchunk_k*nb_floats_reg, m); ++f)
      { 
        const int frel = f % nb_floats_reg;

        auto ymm_i = ymm_c[ireg];
        auto xmm_i = ymm_c[ireg].half();

        // load scatter index (i,j)+f
        assembler.mov(regs::edx, dword_ptr(regs::rax, f*SSIZEOF(int)));
        assembler.cdqe(regs::edx);

        // compute address to C element 
        assembler.mov(regs::rsi, qword_ptr(regs::rbp, stack_c));
        assembler.lea(regs::rdx, Mem{regs::rsi, regs::rdx, 2, 0});

        // shift values in register to the right
        assembler.vpalignr(ymm_mask, xmm_i, xmm_i, Imm{frel*sizeof(float)});

        // store element in C
        assembler.vmovss(dword_ptr(regs::rdx), ymm_mask.half());

        // move higher part to lower part for extraction
        if (f == 3)
        {
          assembler.vextractf128(xmm_i, ymm_i, Imm{1});
        }
      }

    }

    assembler.add(dword_ptr(regs::rbp, stack_iloop), Imm{nb_reg_vchunk_m*nb_floats_reg});
    assembler.cmp(dword_ptr(regs::rbp, stack_iloop), m);
    assembler.jl(iloop_label);

  } // ======== End loop over M chunks ===============

  assembler.inc(dword_ptr(regs::rbp, stack_jloop));
  assembler.cmp(dword_ptr(regs::rbp, stack_jloop), n);
  assembler.jl(jloop_label);

  // return zero
  assembler.mov(regs::eax, dword_ptr(regs::rbp, stack_jloop));

  assembler.pop(regs::rbp);

  assembler.ret();

  // code
  int (*fn)(float*,float*,float*);
                 
  asmjit::Error err = m_jit_runtime.add(&fn, &code);   
  if (err) 
  {
    std::string err_msg = fmt::format("Failed to compile kernel: {}", err);
    throw std::runtime_error(err_msg);
  }

  return KernelFunctionSp(fn);

}
#endif 

KernelFunctionSp KernelImpl::create_kernel_xmm_float32()
{
  using namespace asmjit;
  using namespace x86;

  CodeHolder code; 
  code.init(m_jit_runtime.environment(), m_jit_runtime.cpuFeatures());

  x86::Assembler assembler(&code);

  /*
   *       c    +=    a a a a  * b
   *       c          a a a a    b
   *       c          a a a a    b
   *       c          a a a a    b
   */

  // Get number of registers along I
  const int m = Utils::ssize(m_info_in1.scatter_row);
  const int n = Utils::ssize(m_info_in2.scatter_col);
  const int k = Utils::ssize(m_info_in1.scatter_col);

  const float alpha = std::any_cast<float>(m_alpha);
  const float beta = std::any_cast<float>(m_beta);

  if (!m_jit_runtime.cpuFeatures().x86().hasAVX2())
  {
    throw std::runtime_error("CPU does not have AVX2 capabilities");
  }

  // ymm register size
  const int regSizeByte = 32;

  // how many units in a register?
  const int nb_floats_reg = regSizeByte/sizeof(float);
  // == const int m_block_size (for now)
  
  const int nb_regs_m = Utils::div_ceil(m, nb_floats_reg);

  // k-block size
  const int k_block_size = 4;

  // number of elements in epilogue for m 
  const int m_epiloop = m % nb_floats_reg;

  // number of elements in main loop for m
  const int m_mainloop = (m/nb_floats_reg) * nb_floats_reg;

  // same, but for k
  const int k_epiloop = k % k_block_size;
  const int k_mainloop = (k/k_block_size) * k_block_size;

  DEBUG_VAR(m_logger, nb_regs_m);
  DEBUG_VAR(m_logger, k_block_size);
  DEBUG_VAR(m_logger, nb_floats_reg);
  DEBUG_VAR(m_logger, m_mainloop);
  DEBUG_VAR(m_logger, m_epiloop);
  DEBUG_VAR(m_logger, k_epiloop);
  DEBUG_VAR(m_logger, k_mainloop);

  enum LoopType
  {
    LOOP_MAIN = 0,
    LOOP_EPI = 1
  };

  std::vector<LoopType> m_loop_types;
  if (m_mainloop > 0)
  {
    m_loop_types.push_back(LOOP_MAIN);
  }
  if (m_epiloop > 0)
  {
    m_loop_types.push_back(LOOP_EPI);
  }

  std::vector<LoopType> k_loop_types;
  if (k_mainloop > 0)
  {
    k_loop_types.push_back(LOOP_MAIN);
  }
  if (k_epiloop > 0)
  {
    k_loop_types.push_back(LOOP_EPI);
  }

  // RDI: address of A
  // RSI: address of B
  // RDX: address of C
  int stack_offset = 0;

  // allocate space for stack variables

  // address to A
  const int stack_a = (stack_offset -= 8);
  // address to B
  const int stack_b = (stack_offset -= 8);
  // address to C
  const int stack_c = (stack_offset -= 8);

  // N loop variable
  const int stack_jloop = (stack_offset -= 4);
  // M loop variable
  const int stack_iloop = (stack_offset -= 4);
  // K loop variable
  const int stack_kloop = (stack_offset -= 4);

  // alpha and beta
  const int stack_alpha = (stack_offset -= 4);
  const int stack_beta = (stack_offset -= 4);

  // // Current position in (scatter) matrix A
  // const int stack_a_pos = (stack_offset -= 4);
  // // Current position in (scatter) matrix B
  // const int stack_b_pos = (stack_offset -= 4);
  // // CUrrent position in (scatter) matrix C
  // const int stack_c_pos = (stack_offset -= 4);

  const int stack_size = Utils::round_next_multiple(-stack_offset, 16);

  Log::debug(m_logger, "Stack: {}\n", stack_size);

  // organize registers

  int regID = 16;
  auto fetch_reg = [&regID]()
  {
    regID--;
    if (regID < 0)
    {
      throw std::runtime_error("No ymm registers available");
    }
    return Ymm(regID);
  };

  // registers to hold block C_Ij
  Ymm ymm_c = fetch_reg(); 

  // registers to hold block A_IK
  std::vector<Ymm> vymm_a(k_block_size,Ymm(0));
  std::generate(vymm_a.begin(), vymm_a.end(), fetch_reg);
  //Ymm ymm_a = fetch_reg();

  Ymm ymm_tmp = fetch_reg();

  Ymm ymm_mask = fetch_reg();

  // register for beta
  Ymm ymm_beta = fetch_reg();

  Ymm ymm_alpha = fetch_reg();

  // ======== PRECOMPUTE OFFSETS IN MEMORY ===========

  m_buffer.resize(((m*k)+(k*n)+(m*n)+2*nb_floats_reg)*sizeof(int));
  int* plast = reinterpret_cast<int*>(m_buffer.data());

  auto scatter_idx_a = Span<int>{plast, m*k};
  plast += scatter_idx_a.size();

  auto scatter_idx_b = Span<int>{plast, k*n};
  plast += scatter_idx_b.size();

  auto scatter_idx_c = Span<int>{plast, m*n};
  plast += scatter_idx_c.size();

  auto mask_values = Span<int>(plast, nb_floats_reg);
  plast += mask_values.size();

  auto p_beta = reinterpret_cast<float*>(plast);
  plast++;

  auto p_alpha = reinterpret_cast<float*>(plast);
  plast++;

  auto compute_scatter_indices = 
  [](auto& _pdest, const auto& _scatter_row, const auto& _scatter_col)
  {
    const int nb_row = Utils::ssize(_scatter_row);
    const int nb_col = Utils::ssize(_scatter_col);

    for (int j = 0; j < nb_col; ++j)
    {
      for (int i = 0; i < nb_row; ++i)
      {
        const int val = _scatter_row[i] + _scatter_col[j];
        if (val >= nb_row*nb_col)
        {
          throw std::runtime_error("Scatter index overflow");
        }
        _pdest[i+j*nb_row] = val;
      }
    }
  };

  compute_scatter_indices(scatter_idx_a, m_info_in1.scatter_row, m_info_in1.scatter_col);
  compute_scatter_indices(scatter_idx_b, m_info_in2.scatter_row, m_info_in2.scatter_col);
  compute_scatter_indices(scatter_idx_c, m_info_out.scatter_row, m_info_out.scatter_col);

  Log::debug(m_logger, "Scatter idx C: {}", 
    fmt::join(scatter_idx_c.begin(), scatter_idx_c.end(), ","));

  // Store mask for vector loading
  std::fill(mask_values.begin(), mask_values.end(), 0);

  for (int f = 0; f < m_epiloop; ++f)
  {
    mask_values[f] = 0xFFFFFFFF;
  }

  *p_alpha = alpha;
  *p_beta = beta;

  // ======== START ASSEMBLING FUNCTION =============

  // put stack pointer into rbp
  assembler.push(regs::rbp);
  assembler.mov(regs::rbp, regs::rsp);

  //assembler.sub(regs::rsp, Imm{stack_size});

  // static const char* formatString = "Hello\n";

  // assembler.mov(regs::eax, Imm{0});
  // assembler.mov(regs::rdi, Imm{(void*)formatString});

  // assembler.call(printf);

  // put addresses of arrays into stack
  assembler.mov(qword_ptr(regs::rbp, stack_a), regs::rdi);
  assembler.mov(qword_ptr(regs::rbp, stack_b), regs::rsi);
  assembler.mov(qword_ptr(regs::rbp, stack_c), regs::rdx);

  // put alpha and beta into stack
  Log::debug(m_logger, "alpha={}, beta={}", alpha, beta);

  // load alpha and beta
  //assembler.vbroadcastss(ymm_alpha, Mem{regs::rbp, stack_alpha});
  assembler.mov(regs::rax, (void*)p_beta);
  assembler.mov(regs::rsi, (void*)p_alpha);
  
  assembler.vbroadcastss(ymm_beta, dword_ptr(regs::rax, 0));
  assembler.vbroadcastss(ymm_alpha, dword_ptr(regs::rsi, 0));

  // C kernel
    // Loop over n: i = 0 ---> n
      // Loop over m: j = 0 ---> m/vecsize * vecsize
        // Load C(I,j)
        // Call K kernel
        // Store C(I,j)
      // end
      // Epilogue loop over m: j (rest)
        // Maskload C(I,j)
        // Call mask K kernel
        // Maskstore C(I,j)
      // end epilogue
    // end loop n
  // end KERNEL

  Log::debug(m_logger, "Address for scatter c: {}", (void*)scatter_idx_c.data());

  // ======= LOOP OVER COLUMNS J ==============  

  // init loop var
  assembler.mov(dword_ptr(regs::rbp, stack_jloop), 0);

  Label jloop_label = assembler.newLabel();
  assembler.bind(jloop_label);

  // init loop var
  assembler.mov(dword_ptr(regs::rbp, stack_iloop), 0);
  Label iloop_label = assembler.newLabel();

  for (auto loop_type : m_loop_types)
  { 
    // ======= LOOP OVER CHUNKS M =========
    Log::debug(m_logger, "Assembling {} (m)", (loop_type == LOOP_MAIN) ? "main loop" : "epilogue");
    
    if (loop_type == LOOP_MAIN)
    {
      assembler.bind(iloop_label);
    }

    // move mask to register
    if (loop_type == LOOP_EPI) 
    { 
      assembler.mov(regs::rax, mask_values.data());
      assembler.vmovdqu(ymm_mask, ymmword_ptr(regs::rax));
    }
    else 
    {
      // set all bits to 1
      assembler.vpcmpeqd(ymm_mask, ymm_mask, ymm_mask);
    }

    // Load chunk C(MREG,j)
    if (!Utils::bit_equal(beta,0.0f))
    {
      // i + j*m => eax
      assembler.mov(regs::eax, Mem{regs::rbp, stack_jloop});
      assembler.imul(regs::eax, m);
      assembler.mov(regs::edx, regs::eax);
      assembler.mov(regs::eax, Mem{regs::rbp, stack_iloop});
      assembler.add(regs::eax, regs::edx);
      assembler.cdqe();

      assembler.mov(regs::rdx, (void*)scatter_idx_c.data());

      // get address to scatter index
      assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, 0});

      // move scatter indices into ymm
      if (loop_type == LOOP_MAIN)
      {
        assembler.vmovdqu(ymm_tmp, ymmword_ptr(regs::rax));
      }
      else 
      {
        assembler.vmaskmovps(ymm_tmp, ymm_mask, ymmword_ptr(regs::rax));
      }

      assembler.mov(regs::rax, qword_ptr(regs::rbp, stack_c));

      // load C register using scatter indices
      assembler.vgatherdps(ymm_c, Mem{regs::rax, ymm_tmp, 2, 0}, ymm_mask);
    
      // C *= beta
      assembler.vmulps(ymm_c, ymm_c, ymm_beta);
    }
    else 
    {
      // beta is zero, so just zero out the registers
      assembler.vpxor(ymm_c, ymm_c, ymm_c);
    }

    // Loop over k
    assembler.mov(dword_ptr(regs::rbp, stack_kloop), 0);
    Label kloop_label = assembler.newLabel();

    for (auto kloop_type : k_loop_types)
    {
      if (Utils::bit_equal(alpha, 0.f))
      {
        break;
      }

      if (kloop_type == LOOP_MAIN)
      {
        assembler.bind(kloop_label);
      }
      
      // i + k*m => eax
      assembler.mov(regs::eax, Mem{regs::rbp, stack_kloop});
      assembler.imul(regs::eax, m);
      assembler.mov(regs::edx, regs::eax);
      assembler.mov(regs::eax, Mem{regs::rbp, stack_iloop});
      assembler.add(regs::eax, regs::edx);
      assembler.cdqe(regs::eax);  

      assembler.mov(regs::rdx, (void*)scatter_idx_a.data());

      // get address to scatter index
      assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, 0});

      // k + j*K => ebx
      assembler.mov(regs::ebx, Mem{regs::rbp, stack_jloop});
      assembler.imul(regs::ebx, k);
      assembler.mov(regs::edx, regs::ebx);
      assembler.mov(regs::ebx, Mem{regs::rbp, stack_kloop});
      assembler.add(regs::ebx, regs::edx);
      assembler.cdqe(regs::ebx);

      assembler.mov(regs::rdx, (void*)scatter_idx_b.data());

      // get address to scatter index
      assembler.lea(regs::rbx, Mem{regs::rdx, regs::rbx, 2, 0});

      assembler.mov(regs::r8, qword_ptr(regs::rbp, stack_a));
      assembler.mov(regs::r9, qword_ptr(regs::rbp, stack_b));
      
      const int nb_kreg = (kloop_type == LOOP_MAIN) ? k_block_size : k_epiloop;
      for (int kreg = 0; kreg < nb_kreg; ++kreg)
      {
        // move scatter indices into ymm (i + k*m + kreg*m)
        if (loop_type == LOOP_MAIN)
        {
          assembler.vmovdqu(ymm_tmp, ymmword_ptr(regs::rax,kreg*m*sizeof(int)));
          assembler.vpcmpeqd(ymm_mask, ymm_mask, ymm_mask);
        }
        else 
        {
          assembler.mov(regs::rsi, mask_values.data());
          assembler.vmovdqu(ymm_mask, ymmword_ptr(regs::rsi));
          assembler.vmaskmovps(ymm_tmp, ymm_mask, ymmword_ptr(regs::rax,kreg*m*sizeof(int)));
        }

        // load A register using scatter indices
        assembler.vgatherdps(vymm_a[kreg], Mem{regs::r8, ymm_tmp, 2, 0}, ymm_mask);

        // load scatter index for b (k + kreg + j*K)
        assembler.mov(regs::edi, dword_ptr(regs::rbx, kreg*sizeof(int)));

        // broadcast value
        assembler.vbroadcastss(ymm_tmp, dword_ptr(regs::r9, regs::edi, 2));

        // mult by alpha
        assembler.vmulps(vymm_a[kreg], vymm_a[kreg], ymm_alpha);

        // C += A*B
        assembler.vfmadd231ps(ymm_c, vymm_a[kreg], ymm_tmp);
      
      }

      if (kloop_type == LOOP_MAIN)
      {
        assembler.add(dword_ptr(regs::rbp, stack_kloop), Imm{k_block_size});
        assembler.cmp(dword_ptr(regs::rbp, stack_kloop), k_mainloop);
        assembler.jl(kloop_label);
      }

    }

    // store chunk C(MREG,j)
    
    // i + j*m => eax
    assembler.mov(regs::eax, Mem{regs::rbp, stack_jloop});
    assembler.imul(regs::eax, m);
    assembler.mov(regs::edx, regs::eax);
    assembler.mov(regs::eax, Mem{regs::rbp, stack_iloop});
    assembler.add(regs::eax, regs::edx);
    assembler.cdqe();

    assembler.mov(regs::rdx, (void*)scatter_idx_c.data());

    // get address to scatter index
    assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, 0});

    // Store elements in ymm to C
    const int reg_size = (loop_type == LOOP_MAIN) ? nb_floats_reg : m_epiloop;

    for (int f = 0; f < reg_size; ++f)
    { 
      const int f_half = f % (nb_floats_reg/2);

      auto xmm_c = ymm_c.half();

      // load scatter index (i,j)+f
      assembler.mov(regs::edx, dword_ptr(regs::rax, f*SSIZEOF(int)));
      assembler.cdqe(regs::edx);

      // compute address to C element 
      assembler.mov(regs::rsi, qword_ptr(regs::rbp, stack_c));
      assembler.lea(regs::rdx, Mem{regs::rsi, regs::rdx, 2, 0});

      if (f_half != 0)
      {
        // shift values in register to the right
        assembler.vpalignr(ymm_tmp, xmm_c, xmm_c, Imm{f_half*sizeof(float)});
        assembler.vmovss(dword_ptr(regs::rdx), ymm_tmp.half());
      }
      else 
      {
        // zero offset, so just move directly
        assembler.vmovss(dword_ptr(regs::rdx), xmm_c);
      }      

      // move higher part to lower part for extraction
      if (f == 3)
      {
        assembler.vextractf128(xmm_c, ymm_c, 1);
      }
    }

    if (loop_type == LOOP_MAIN)
    {
      assembler.add(dword_ptr(regs::rbp, stack_iloop), Imm{nb_floats_reg});
      assembler.cmp(dword_ptr(regs::rbp, stack_iloop), m_mainloop);
      assembler.jl(iloop_label);
    }

  } // end for loop_type

  // ======== End loop over M chunks ===============
  assembler.inc(dword_ptr(regs::rbp, stack_jloop));
  assembler.cmp(dword_ptr(regs::rbp, stack_jloop), n);
  assembler.jl(jloop_label);

  // return zero
  assembler.mov(regs::eax, 0);

  assembler.pop(regs::rbp);

  assembler.ret();

  // code
  int (*fn)(float*,float*,float*);
                 
  asmjit::Error err = m_jit_runtime.add(&fn, &code);   
  if (err) 
  {
    std::string err_msg = fmt::format("Failed to compile kernel: {}", err);
    throw std::runtime_error(err_msg);
  }

  return KernelFunctionSp(fn);

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