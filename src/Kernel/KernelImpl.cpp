#include "KernelImpl.h"
#include "AvxHelper.h"
#include "LAPACK.h"
#include "BlockSpan.h"

namespace Shtensor 
{

KernelImpl::KernelImpl(const std::string _expr, 
                       const std::vector<int>& _sizes_in1, 
                       const std::vector<int>& _sizes_in2,
                       const std::vector<int>& _sizes_out,
                       std::any _alpha, 
                       std::any _beta,
                       FloatType _float_type,
                       KernelType _kernel_type)
  : m_expression(_expr)
  , m_sizes_in1(_sizes_in1)
  , m_sizes_in2(_sizes_in2)
  , m_sizes_out(_sizes_out)
  , m_alpha(_alpha)
  , m_beta(_beta)
  , m_float_type(_float_type)
  , m_kernel_type(_kernel_type)
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

  constexpr auto cenum = [](FloatType _type, KernelType _method)
  {
    return static_cast<int>(_method)*10 + static_cast<int>(_type);
  };

  int full_type = cenum(m_float_type, m_kernel_type);
  
  switch (full_type)
  {
    case cenum(FloatType::FLOAT32, KernelType::LAPACK):
    {
      m_kernel_function = create_kernel_lapack_float32();
      break;
    }
    case cenum(FloatType::FLOAT64, KernelType::LAPACK):
    {
      m_kernel_function = create_kernel_lapack_float64();
      break;
    }
    case cenum(FloatType::FLOAT32, KernelType::XMM):
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
  auto func = [p_buffer_a,p_buffer_b,p_buffer_c,size_a,size_b,size_c,this]
    (float* _a, float* _b, float* _c, int64_t _nb_ops) -> int
  {
    const std::vector<int> order_a = Utils::concat(m_info_in1.map_row,m_info_in1.map_col);
    const std::vector<int> order_b = Utils::concat(m_info_in2.map_row,m_info_in2.map_col);
    const std::vector<int> order_c = Utils::concat(m_info_out.map_row,m_info_out.map_col);

    for (int64_t iop = 0; iop < _nb_ops; ++iop)
    {
      float* pA = _a + iop*size_a;
      float* pB = _b + iop*size_b;
      float* pC = _c + iop*size_c;

      Utils::reshape(pA, m_sizes_in1, order_a, p_buffer_a);
      Utils::reshape(pB, m_sizes_in2, order_b, p_buffer_b);
      Utils::reshape(pC, m_sizes_out, order_c, p_buffer_c);

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

      Utils::reshape(p_buffer_c, sizes_c_r, order_c_r, pC);
    }

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
              (double* _a, double* _b, double* _c, int64_t _nb_ops) mutable -> int
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

  // check if m is contiguous
  auto is_contig = [](const auto& _map) -> bool
  {
    auto row_info = _map;
    auto row_info_sorted = row_info;
    std::sort(row_info_sorted.begin(), row_info_sorted.end());

    return row_info == row_info_sorted 
      && row_info.front() == 0 
      && row_info.back() == Utils::ssize(row_info)-1;
  };

  const auto arow_contig = is_contig(m_info_in1.map_row);

  const auto crow_contig = is_contig(m_info_out.map_row);

  DEBUG_VAR(m_logger, arow_contig);

  DEBUG_VAR(m_logger, crow_contig);

  const float alpha = std::any_cast<float>(m_alpha);
  const float beta = std::any_cast<float>(m_beta);

  if (!m_jit_runtime.cpuFeatures().x86().hasAVX2())
  {
    throw std::runtime_error("CPU does not have AVX2 capabilities");
  }

  // register size
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

  AvxHelper helper{AvxHelper::AVX2, FloatType::FLOAT32, assembler};

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

  // N loop variable
  const int stack_jloop = (stack_offset -= 8);
  // M loop variable
  const int stack_iloop = (stack_offset -= 8);
  // K loop variable
  const int stack_kloop = (stack_offset -= 8);

  // macro loop
  const int stack_oploop = (stack_offset -= 8);
  const int stack_nbops = (stack_offset -= 8);

  // // Current position in (scatter) matrix A
  // const int stack_a_pos = (stack_offset -= 4);
  // // Current position in (scatter) matrix B
  // const int stack_b_pos = (stack_offset -= 4);
  // // CUrrent position in (scatter) matrix C
  // const int stack_c_pos = (stack_offset -= 4);

  const int stack_size = Utils::round_next_multiple(-stack_offset, 16);

  Log::debug(m_logger, "Stack: {}\n", stack_size);

  // organize registers

  uint32_t regID = 0;
  auto fetch_reg = [&regID]()
  {
    regID++;
    if (regID >= 16)
    {
      throw std::runtime_error("No vector registers available");
    }
    return regID;
  };

  // registers to hold block C_Ij
  uint32_t vecid_c = fetch_reg(); 

  // registers to hold block A_IK
  uint32_t vecid_a = fetch_reg();

  uint32_t vecid_tmp0 = fetch_reg();

  uint32_t vecid_tmp1 = fetch_reg();

  uint32_t vecid_mask = fetch_reg();

  uint32_t vecid_idx = fetch_reg();

  uint32_t vecid_b = fetch_reg();

  uint32_t vecid_beta = fetch_reg();

  uint32_t vecid_alpha = fetch_reg();

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
  assembler.push(regs::rbx);
  assembler.push(regs::r12);
  assembler.push(regs::r13);
  assembler.push(regs::r14);
  assembler.push(regs::r15);

  assembler.mov(regs::rbp, regs::rsp);

  // rax - 
  // rbx - 
  // rcx - 
  // rdx 
  // rdi
  // rsi
  // r8  - A
  // r9  - B
  // r10 - C
  // r11 - scata
  // r12 - scatb
  // r13 - scatc
  // r14 - reserved for use in avxhelper
  // r15 - reserved for use in avxhelper

  auto reg_addr_a = regs::r8;
  auto reg_addr_b = regs::r9;
  auto reg_addr_c = regs::r10;

  // put addresses of arrays into stack
  assembler.mov(reg_addr_a, regs::rdi);
  assembler.mov(reg_addr_b, regs::rsi);
  assembler.mov(reg_addr_c, regs::rdx);

  // put alpha and beta into stack
  Log::debug(m_logger, "alpha={}, beta={}", alpha, beta);

  // load alpha and beta
  assembler.mov(regs::rax, (void*)p_beta);
  assembler.mov(regs::rsi, (void*)p_alpha);

  helper.broadcast(vecid_beta, Mem{regs::rax, 0});
  helper.broadcast(vecid_alpha, Mem{regs::rsi, 0});
  
  assembler.mov(qword_ptr(regs::rbp, stack_nbops), regs::rcx);

  // load mask
  assembler.mov(regs::rax, mask_values.data());
  helper.load_mask(vecid_mask, regs::rax);

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

  // ======= LOOP OVER OPERATIONS ==============
  assembler.mov(qword_ptr(regs::rbp, stack_oploop), 0);

  Label oploop_label = assembler.newLabel();
  assembler.bind(oploop_label);

  assembler.prefetch(Mem{reg_addr_a,0});
  assembler.prefetch(Mem{reg_addr_b,0});
  assembler.prefetch(Mem{reg_addr_c,0});

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

    const bool i_is_epilogue = (loop_type == LOOP_EPI);

    Log::debug(m_logger, "Assembling {} (m)", (loop_type == LOOP_MAIN) ? "main loop" : "epilogue");
    
    if (loop_type == LOOP_MAIN)
    {
      assembler.bind(iloop_label);
    }

    // move mask to register
    helper.copy_mask(vecid_tmp1, (loop_type == LOOP_EPI), vecid_mask);

    // helper.load_mask(0, 1);

    // Load chunk C(MREG,j)
    if (!Utils::bit_equal(beta,0.0f))
    {
      // i + j*m => eax
      assembler.mov(regs::eax, Mem{regs::rbp, stack_jloop});
      assembler.mov(regs::ebx, Mem{regs::rbp, stack_iloop});
      assembler.imul(regs::eax, m);
      assembler.add(regs::eax, regs::ebx);
      assembler.cdqe(regs::eax);

      assembler.mov(regs::rbx, (void*)scatter_idx_c.data());

      // load indices into tmp1
      helper.load_indices(vecid_idx, i_is_epilogue, crow_contig, regs::rbx, regs::rax, vecid_tmp1);

      // load tensor using indices and mask
      helper.load_tensor(vecid_c, i_is_epilogue, crow_contig, reg_addr_c, regs::rax, vecid_tmp1, vecid_idx);

      // C *= beta
      helper.mul(vecid_c, vecid_c, vecid_beta);

    }
    else 
    {
      // beta is zero, so just zero out the registers
      helper.zero(vecid_c);
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

      assembler.mov(regs::rcx, (void*)scatter_idx_a.data());

      // if (!arow_contig)
      // {
      //   // get address to scatter index
      //   assembler.mov(regs::rdx, (void*)scatter_idx_a.data());
      //   assembler.lea(regs::rax, Mem{regs::rdx, regs::rax, 2, 0});
      // }
      // else
      // {
      //   // get address to a
      //   assembler.lea(regs::rax, Mem{reg_addr_a, regs::rax, 2, 0});
      // }

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
      
      const int nb_kreg = (kloop_type == LOOP_MAIN) ? k_block_size : k_epiloop;
      for (int kreg = 0; kreg < nb_kreg; ++kreg)
      {
        // load scatter index for b (k + kreg + j*K)
        assembler.mov(regs::edi, dword_ptr(regs::rbx, kreg*sizeof(int)));

        helper.copy_mask(vecid_tmp1, i_is_epilogue, vecid_mask);

        helper.load_indices(vecid_tmp0, i_is_epilogue, arow_contig, regs::rcx, regs::rax, vecid_tmp1);

        helper.load_tensor(vecid_a, i_is_epilogue, arow_contig, reg_addr_a, regs::rax, vecid_tmp1, vecid_tmp0);

        // broadcast value of b to whole vector
        helper.broadcast(vecid_b, Mem{reg_addr_b, regs::edi, 2, 0});

        // mult by alpha
        helper.mul(vecid_a, vecid_a, vecid_alpha);

        // C += A*B
        helper.fmadd(vecid_c, vecid_a, vecid_b);

        assembler.add(regs::rax, m);
      
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

    helper.copy_mask(vecid_tmp1, i_is_epilogue, vecid_mask);

    assembler.mov(regs::rbx, scatter_idx_c.data());

    helper.load_indices(vecid_tmp0, i_is_epilogue, crow_contig, regs::rbx, regs::rax, vecid_tmp1);

    helper.store_tensor(i_is_epilogue, crow_contig, reg_addr_c, regs::rax, vecid_tmp1, vecid_tmp0, vecid_c, m_epiloop);

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

  // ======== End loop over ops ===============

  assembler.add(reg_addr_a, m*k*sizeof(float));
  assembler.add(reg_addr_b, k*n*sizeof(float));
  assembler.add(reg_addr_c, m*n*sizeof(float));

  assembler.inc(qword_ptr(regs::rbp, stack_oploop));
  assembler.mov(regs::rax, qword_ptr(regs::rbp, stack_nbops));
  assembler.cmp(qword_ptr(regs::rbp, stack_oploop), regs::rax);
  assembler.jl(oploop_label);

  // return zero
  assembler.mov(regs::eax, 0);

  assembler.pop(regs::r15);
  assembler.pop(regs::r14);
  assembler.pop(regs::r13);
  assembler.pop(regs::r12);
  assembler.pop(regs::rbx);
  assembler.pop(regs::rbp);

  assembler.ret();
                 
  asmjit::Error err = m_jit_runtime.add(&m_xmm_fn_holder, &code);   

  // std::vector<uint8_t> code_data(code.codeSize(),0);
  // code.copyFlattenedData(code_data.data(), code_data.size());

  // fmt::print("\\x{:x}\n", fmt::join(code_data.begin(), code_data.end(), "\\x"));

  if (err) 
  {
    std::string err_msg = fmt::format("Failed to compile kernel: {}", err);
    throw std::runtime_error(err_msg);
  }

  return [this](float* _a, float* _b, float* _c, int64_t _nb_ops) 
    { 
      return this->m_xmm_fn_holder(_a, _b, _c, _nb_ops);
    };

}


std::string KernelImpl::get_info()
{
  std::string out;

  out += fmt::format("Kernel Info\n");
  out += fmt::format("  Type: {}\n", static_cast<int>(m_kernel_type));
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

} // namespace Shtensor