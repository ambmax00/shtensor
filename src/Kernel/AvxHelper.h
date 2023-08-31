#ifndef SHTENSOR_AVXHELPER
#define SHTENSOR_AVXHELPER

#include "Utils.h"
#include "asmjit/x86.h"

namespace Shtensor
{

class AvxHelper 
{
 public:

  using regdw = decltype(asmjit::x86::regs::rax);

  using regw = decltype(asmjit::x86::regs::eax);

  using Assembler = asmjit::x86::Assembler;

  using Ymm = decltype(asmjit::x86::regs::ymm0);

  enum Type
  {
    AVX2 = 0
  };

  AvxHelper(Type _avx_type, FloatType _float_type, Assembler& _assembler);  

  void load_mask(uint32_t _vecidx_mask, regdw _mask_address);

  void copy_mask(uint32_t _vecidx_copy, bool _is_epilogue, uint32_t _vecidx_mask);

  void load_indices(uint32_t _vecidx_indices,
                    bool _is_epilogue, 
                    bool _contiguous, 
                    regdw _reg_scatter, 
                    regdw _reg_index,
                    uint32_t _vecidx_mask);

  void load_tensor(uint32_t _vecidx_tensor,
                   bool _is_epilogue,
                   bool _contiguous,
                   regdw _reg_addr_tensor, 
                   regdw _reg_index,
                   uint32_t _vecidx_mask, 
                   uint32_t _vecidx_indices);

  void mul(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t vecidx_b);

  void broadcast(uint32_t _vecidx, regdw _reg_addr_tensor, regw _reg_offset);

  void fmadd(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b);

 private:

  void load_mask_avx2(uint32_t _vecidx_mask, regdw _mask_address);

  void copy_mask_avx2(uint32_t _vecid_copy, bool _is_epilogue, uint32_t _vecid_mask);

  void load_indices_avx2(uint32_t _vecidx_indices,
                         bool _is_epilogue,
                         regdw _reg_scatter, 
                         regdw _reg_index,
                         uint32_t _vecidx_mask);

  void load_tensor_avx2_noncontig(uint32_t _vecidx_tensor,
                                  regdw _reg_addr_tensor, 
                                  uint32_t _vecidx_mask, 
                                  uint32_t _vecidx_indices);

  void load_tensor_avx2_contig(uint32_t _vecidx_tensor,
                               bool _is_epilogue,
                               regdw _reg_addr_tensor, 
                               regdw _reg_index,
                               uint32_t _vecidx_mask);

  void mul_avx2(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b);

  void broadcast_avx2(uint32_t _vecidx, regdw _reg_addr_tensor, regw _reg_offset);

  void fmadd_avx2(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b);

  Type m_avx_type;

  FloatType m_float_type;

  Assembler& m_assembler;


};

} // namespace shtensor

#endif