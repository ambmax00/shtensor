#ifndef SHTENSOR_BASETENSOR_H
#define SHTENSOR_BASETENSOR_H

#include "Context.h"
#include "Definitions.h"
#include "Grid.h"
#include "ShmemInterface.h"

namespace Shtensor
{

#if 1
class BaseTensor
{
 public:

  template <typename T>
  using VVector = std::vector<std::vector<T>>;

  struct SparseInfo
  {
    Window<int64_t> rows;
    Window<int64_t> indices;
    Window<int64_t> offsets;
  };

  BaseTensor(const Context& m_ctx, 
             const VVector<int>& _block_sizes,
             int N, FloatType _type);

  ~BaseTensor()
  {
  }

  void reserve_all();

  void reserve(const std::vector<int>* _p_idx_begin);

  void compress();

  void print_info();

  int64_t get_max_block_size();

  double get_occupation();

  const VVector<int>& get_local_indices() const { return m_local_block_idx; }

  const SparseInfo& get_sparse_info() const { return m_sparse_info; }

  int64_t get_nb_nzblocks_local() { return m_sinfo_local.nb_nzblocks; }

  int64_t get_nb_nzblocks_global() { return m_sinfo_global.nb_nzblocks; }

 protected:

  // collect all block information from all nodes (collective)
  void sync_sinfo_global();

  Context m_ctx;

  int m_dim;

  FloatType m_float_type;

  VVector<int> m_block_sizes;

  Grid m_grid;
  
  VVector<int> m_block_coords;

  std::vector<int64_t> m_block_dims;

  std::vector<int64_t> m_block_strides;

  VVector<int> m_local_block_idx;
  
  // Distribution m_distribution;

  struct SizeInfo 
  {
    int64_t nb_elements;
    int64_t nb_blocks;
    int64_t nb_nze;
    int64_t nb_nze_sym;
    int64_t nb_nzblocks;
    int64_t nb_nzblocks_sym;
  };

  SizeInfo m_sinfo_local;

  SizeInfo m_sinfo_global;
  
  SparseInfo m_sparse_info;

  Window<uint8_t> m_win_data;

  Log::Logger m_logger;

};
#endif

} // namespace Shtensor

#endif