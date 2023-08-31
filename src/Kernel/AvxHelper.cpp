#include "AvxHelper.h"

namespace Shtensor 
{

using namespace asmjit::x86;
using namespace asmjit::x86::regs;

AvxHelper::AvxHelper(Type _avx_type, FloatType _float_type, Assembler& _assembler)
  : m_avx_type(_avx_type)
  , m_float_type(_float_type)
  , m_assembler(_assembler)
{
}  

void AvxHelper::load_mask(uint32_t _vecidx_mask, regdw _mask_address)
{
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      load_mask_avx2(_vecidx_mask, _mask_address);
      break;
    }
  }
}

void AvxHelper::copy_mask(uint32_t _vecidx_copy, bool _is_epilogue, uint32_t _vecidx_mask)
{
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      copy_mask_avx2(_vecidx_copy, _is_epilogue, _vecidx_mask);
      break;
    }
  }
}

void AvxHelper::load_indices(uint32_t _vecidx_indices,
                             bool _is_epilogue, 
                             bool _contiguous, 
                             regdw _reg_scatter, 
                             regdw _reg_index,
                             uint32_t _vecidx_mask)
{
  // no need to load indices for contiguous case
  if (_contiguous)
  {
    return;
  }

  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      load_indices_avx2(_vecidx_indices, _is_epilogue, _reg_scatter, _reg_index, _vecidx_mask);
      break;
    }
  }

}

void AvxHelper::load_tensor(uint32_t _vecidx_tensor,
                            bool _is_epilogue,
                            bool _contiguous,
                            regdw _reg_addr_tensor, 
                            regdw _reg_index,
                            uint32_t _vecidx_mask, 
                            uint32_t _vecidx_indices)
{ 
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      if (_contiguous)
      {
        load_tensor_avx2_contig(_vecidx_tensor, _is_epilogue, _reg_addr_tensor, _reg_index, _vecidx_mask);
      }
      else 
      {
        load_tensor_avx2_noncontig(_vecidx_tensor, _reg_addr_tensor, _vecidx_mask, _vecidx_indices);
      }
      break;
    }
  }
}

void AvxHelper::mul(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t vecidx_b)
{
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      mul_avx2(_vecidx_result, _vecidx_a, vecidx_b);
      break;
    }
  }
}

void AvxHelper::broadcast(uint32_t _vecidx, regdw _reg_addr_tensor, regw _reg_offset)
{
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      broadcast_avx2(_vecidx, _reg_addr_tensor, _reg_offset);
      break;
    }
  }
}

void AvxHelper::fmadd(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b)
{
  switch (m_avx_type)
  {
    case Type::AVX2:
    {
      fmadd_avx2(_vecidx_result, _vecidx_a, _vecidx_b);
      break;
    }
  }
}

void AvxHelper::load_mask_avx2(uint32_t _vecidx_mask, regdw _mask_address)
{
  if (m_float_type == FloatType::FLOAT32)
  {
    m_assembler.vmovdqu(Ymm(_vecidx_mask), ymmword_ptr(_mask_address));
  }
}

void AvxHelper::copy_mask_avx2(uint32_t _vecid_copy, bool _is_epilogue, uint32_t _vecid_mask)
{
  auto ymm_copy = Ymm{_vecid_copy};
  auto ymm_mask = Ymm{_vecid_mask};

  if (_is_epilogue) 
  { 
    m_assembler.vmovdqu(ymm_copy, ymm_mask);
  }
  else 
  {
    // set all bits to 1
    m_assembler.vpcmpeqd(ymm_copy, ymm_copy, ymm_copy);
  }
}

void AvxHelper::load_indices_avx2(uint32_t _vecidx_indices,
                                  bool _is_epilogue,
                                  regdw _reg_scatter, 
                                  regdw _reg_index,
                                  uint32_t _vecidx_mask)
{
  auto ymm_mask = Ymm{_vecidx_mask};
  auto ymm_idx = Ymm{_vecidx_indices};

  m_assembler.lea(regs::r14, Mem{_reg_scatter, _reg_index, 2, 0});

  // move scatter indices into ymm
  if (!_is_epilogue)
  {
    if (m_float_type == FloatType::FLOAT32)
    {
      m_assembler.vmovdqu(ymm_idx, ymmword_ptr(regs::r14));
    }
  }
  else 
  {
    if (m_float_type == FloatType::FLOAT32)
    {
      m_assembler.vmaskmovps(ymm_idx, ymm_mask, ymmword_ptr(regs::r14));
    }
    
  }
}

void AvxHelper::load_tensor_avx2_noncontig(uint32_t _vecidx_tensor,
                                           regdw _reg_addr_tensor, 
                                           uint32_t _vecidx_mask, 
                                           uint32_t _vecidx_indices)
{
  if (m_float_type == FloatType::FLOAT32)
  {
    m_assembler.vgatherdps(Ymm(_vecidx_tensor), 
                            Mem{_reg_addr_tensor, Ymm(_vecidx_indices), 2, 0}, 
                            Ymm(_vecidx_mask));
  }
}

void AvxHelper::load_tensor_avx2_contig(uint32_t _vecidx_tensor,
                                        bool _is_epilogue,
                                        regdw _reg_addr_tensor, 
                                        regdw _reg_index,
                                        uint32_t _vecidx_mask)
{
  if (m_float_type == FloatType::FLOAT32)
  {
    // get address to C
    m_assembler.lea(regs::r14, Mem{_reg_addr_tensor, _reg_index, 2, 0});

    if (!_is_epilogue)
    {
      m_assembler.vmovdqu(Ymm(_vecidx_tensor), ymmword_ptr(regs::r14));
    }
    else 
    {
      m_assembler.vmaskmovps(Ymm(_vecidx_tensor), Ymm(_vecidx_mask), ymmword_ptr(regs::r14));
    }
  }
}

void AvxHelper::mul_avx2(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b)
{
  auto ymm_a = Ymm{_vecidx_a};
  auto ymm_b = Ymm{_vecidx_b};
  auto ymm_r = Ymm{_vecidx_result};

  if (m_float_type == FloatType::FLOAT32)
  {
    m_assembler.vmulps(ymm_r, ymm_a, ymm_b);
  }
  
}

void AvxHelper::broadcast_avx2(uint32_t _vecidx, regdw _reg_addr_tensor, regw _reg_offset)
{
  if (m_float_type == FloatType::FLOAT32)
  {
    m_assembler.vbroadcastss(Ymm{_vecidx}, dword_ptr(_reg_addr_tensor, _reg_offset, 2));
  }
}

void AvxHelper::fmadd_avx2(uint32_t _vecidx_result, uint32_t _vecidx_a, uint32_t _vecidx_b)
{
  if (m_float_type == FloatType::FLOAT32)
  {
    m_assembler.vfmadd231ps(Ymm{_vecidx_result}, Ymm{_vecidx_a}, Ymm{_vecidx_b});
  }
}

} // namespace Shtensor