#include "BaseTensor.h"

namespace Shtensor
{

#if 1
BaseTensor::BaseTensor(const Context& _ctx, 
                       const VVector<int>& _block_sizes,
                       int N, FloatType _type)
  : m_ctx(_ctx)
  , m_dim(N)
  , m_float_type(_type)
  , m_block_sizes(_block_sizes)
  , m_grid{_ctx.get_comm(), N}
  , m_block_coords(m_dim)
  , m_block_dims(m_dim)
  , m_block_strides(m_dim)
  , m_local_block_idx(m_dim)
  , m_sinfo_local{}
  , m_sinfo_global{}
  , m_sparse_info{}
  , m_win_data{}
  , m_logger(Log::create("BaseTensor"))
{
  // create block distributions
  for (int i = 0; i < m_dim; ++i)
  {
    m_block_coords[i] = Utils::compute_default_dist(m_block_sizes[i].size(), 
                                                    m_grid.get_grid_dims()[i], 
                                                    m_block_sizes[i]);
  }

  for (int i = 0; i < m_dim; ++i)
  {
    m_block_dims[i] = Utils::ssize(m_block_sizes[i]);
  }

  m_block_strides = Utils::compute_strides(m_block_dims);

  // get local indices
  for (int idim = 0; idim < m_dim; ++idim)
  {
    for (int iblk = 0; iblk < m_block_dims[idim]; ++iblk)
    {
      if (m_block_coords[idim][iblk] == m_grid.get_coords()[idim])
      {
        m_local_block_idx[idim].push_back(iblk);
      }
    }
  }

  // total number of local elements
  Utils::loop_idx(m_local_block_idx,
    [this](const std::vector<int>& _idx)
    {
      int64_t blk_size = 1;
      for (int idim = 0; idim < m_dim; ++idim)
      {
        blk_size *= m_block_sizes[idim][m_local_block_idx[idim][_idx[idim]]];
      } 
      m_sinfo_local.nb_elements += blk_size;
      m_sinfo_local.nb_blocks += 1;
    });

  //sync_sinfo_global();
  MPI_Allreduce(&m_sinfo_local.nb_blocks, &m_sinfo_global.nb_blocks, 1, MPI_INT64_T, MPI_SUM, 
                m_ctx.get_comm());

  MPI_Allreduce(&m_sinfo_local.nb_elements, &m_sinfo_global.nb_elements, 1, MPI_INT64_T, MPI_SUM, 
                m_ctx.get_comm());

}

void BaseTensor::sync_sinfo_global()
{

  // MPI_Allreduce(&m_sinfo_local.nb_blocks, &m_sinfo_global.nb_blocks, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

  // MPI_Allreduce(&m_sinfo_local.nb_elements, &m_sinfo_global.nb_elements, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

  // MPI_Allreduce(&m_sinfo_local.nb_nzblocks, &m_sinfo_global.nb_nzblocks, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

  // MPI_Allreduce(&m_sinfo_local.nb_nzblocks_sym, &m_sinfo_global.nb_nzblocks_sym, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

  // MPI_Allreduce(&m_sinfo_local.nb_nze, &m_sinfo_global.nb_nze, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

  // MPI_Allreduce(&m_sinfo_local.nb_nze_sym, &m_sinfo_global.nb_nze_sym, 1, MPI_INT64_T, MPI_SUM, 
  //               m_ctx.get_comm());

}

void BaseTensor::reserve_all()
{
  const int64_t nb_blocks_local = Utils::varray_mult_ssize(m_local_block_idx);

  VVector<int> indices(m_dim, std::vector<int>(nb_blocks_local));

  Utils::loop_idx(m_local_block_idx, 
    [&indices,this,iloop=0](const std::vector<int>& loop_idx) mutable
    {
      for (int idim = 0; idim < m_dim; ++idim)
      {
        indices[idim][iloop] = m_local_block_idx[idim][loop_idx[idim]];
      }
      iloop++;
    });

  reserve(indices.data());

}

void BaseTensor::reserve(const std::vector<int>* _p_idx_begin)
{
  bool equalSize = std::equal(_p_idx_begin+1, _p_idx_begin+m_dim, _p_idx_begin, 
    [](const auto& _idx0, const auto& _idx1)
    {
      return _idx0.size() == _idx1.size();
    });

  if (!equalSize)
  {
    throw std::runtime_error("reserve: Indices do not have the same size in each dimension");
  }

  const int64_t nb_rows = Utils::ssize(m_block_sizes[0]);
  const int64_t nb_blocks = Utils::ssize(_p_idx_begin[0]);

  m_sinfo_local.nb_nzblocks = nb_blocks;

  MPI_Allreduce(&m_sinfo_local.nb_nzblocks, &m_sinfo_local.nb_nzblocks_sym, 1, MPI_INT64_T, 
                MPI_MAX, m_ctx.get_comm());

  // count slices in each row and the number of total elements in that row
  std::vector<int64_t> nb_slices_row(nb_rows, 0);
  std::vector<int64_t> nb_elements_row(nb_rows, 0);

  for (int64_t iblk = 0; iblk < nb_blocks; ++iblk)
  {
    const int64_t irow = _p_idx_begin[0][iblk];
    nb_slices_row[irow]++;

    int block_size = 1;

    for (auto idim = 0ul; idim < m_dim; ++idim)
    {
      block_size *= m_block_sizes[idim][_p_idx_begin[idim][iblk]];
    }

    nb_elements_row[irow] += block_size;
  }

  ShmemInterface shmem{m_ctx};

  // count number of total elements
  m_sinfo_local.nb_nze = std::accumulate(nb_elements_row.begin(), nb_elements_row.end(), 0);

  m_sparse_info.rows = shmem.allocate<int64_t>(nb_rows);

  m_sparse_info.indices = shmem.allocate<int64_t>(m_sinfo_local.nb_nzblocks_sym);

  m_sparse_info.offsets = shmem.allocate<int64_t>(m_sinfo_local.nb_nzblocks_sym);

  std::copy(nb_slices_row.begin(), nb_slices_row.end(), m_sparse_info.rows.begin());

  std::vector<int64_t> rolled_idx(nb_blocks, 0);

  for (int iblk = 0; iblk < nb_blocks; ++iblk)
  {
    for (int idim = 0; idim < m_dim; ++idim)
    {
      rolled_idx[iblk] += _p_idx_begin[idim][iblk]*m_block_strides[idim];
    } 
  }

  // sort indices
  std::vector<int64_t> perm(nb_blocks);
  std::iota(perm.begin(), perm.end(), 0);

  std::sort(perm.begin(), perm.end(), 
    [&rolled_idx](int64_t val0, int64_t val1)
    {
      return (rolled_idx[val0] < rolled_idx[val1]);
    });

  std::vector<int64_t> sorted_rolled_idx(rolled_idx.size());
  std::generate(sorted_rolled_idx.begin(), sorted_rolled_idx.end(),
    [&perm,&rolled_idx,i=0]() mutable
    {
      return rolled_idx[perm[i++]];
    });

  // copy block indices to sparse info array
  int64_t offset = 0;

  for (int64_t irow = 0; irow < nb_rows; ++irow)
  {
    m_sparse_info.rows[irow] = offset;
    std::copy(sorted_rolled_idx.begin()+offset, 
              sorted_rolled_idx.begin()+offset+nb_slices_row[irow], 
              m_sparse_info.indices.begin()+offset);

    offset += nb_slices_row[irow];
  }

  offset = 0;
  int64_t iblk = 0;

  // go through each block and record offset
  for (int64_t irow = 0; irow < nb_rows; ++irow)
  {
    for (int64_t islice = 0; islice < nb_slices_row[irow]; ++islice)
    {
      std::vector<int> indices(m_dim);
      Utils::unroll_index(m_block_strides, m_dim, m_sparse_info.indices[iblk], indices);

      int blk_size = 1;  
      for (int idim = 0; idim < m_dim; ++idim)
      {
        blk_size *= m_block_sizes[idim][indices[idim]];
      }

      m_sparse_info.offsets[iblk] = offset;

      offset += blk_size;
      iblk++;
    }
  }

  // allocate full space and set to zero
  MPI_Allreduce(&m_sinfo_local.nb_nze, &m_sinfo_local.nb_nze_sym, 1, MPI_INT64_T, MPI_MAX, 
                m_ctx.get_comm());

  m_win_data = shmem.allocate<uint8_t>(m_sinfo_local.nb_nze_sym*Utils::float_size(m_float_type));
  std::fill(m_win_data.begin(), m_win_data.end(), 0);

  // collect global info
  MPI_Allreduce(&m_sinfo_local.nb_nzblocks, &m_sinfo_global.nb_nzblocks, 1, MPI_INT64_T, MPI_SUM, 
                m_ctx.get_comm());

  MPI_Allreduce(&m_sinfo_local.nb_nze, &m_sinfo_global.nb_nze, 1, MPI_INT64_T, MPI_SUM, 
                m_ctx.get_comm());

  m_sinfo_global.nb_nzblocks_sym = m_ctx.get_size() * m_sinfo_local.nb_nzblocks_sym;
  m_sinfo_global.nb_nze_sym = m_ctx.get_size() * m_sinfo_local.nb_nze_sym;
  
}

void BaseTensor::compress()
{
  // loop over blocks and compress
  int prev_row = 0;
  
  bool prev_was_empty = false;

  int64_t chunk_start = 0;
  int64_t chunk_dest = 0;
  int64_t chunk_size = 0;

  int64_t nb_zero_blocks = 0;

  int64_t nb_nze_new = 0;
  int64_t nb_nzblocks_new = 0;

  const int float_size = Utils::float_size(m_float_type);

  for (int64_t iblk = 0; iblk < m_sinfo_local.nb_nzblocks; ++iblk)
  {
    // get current row 
    const int64_t blk_idx = m_sparse_info.indices[iblk];

    const int64_t abs_blk_idx = (blk_idx >= 0) ? blk_idx : -(blk_idx+1);

    std::vector<int> indices(m_dim);
    Utils::unroll_index(m_block_strides, m_dim, abs_blk_idx, indices);
    const int row = indices[0];

    // get block size
    int blk_size = 0;

    for (int i = 0; i < m_dim; ++i)
    {
      blk_size *= m_block_sizes[i][indices[i]];
    }

    const int64_t blk_off = m_sparse_info.offsets[iblk];
    const bool is_empty = (blk_idx < 0);

    if (!is_empty)
    {
      // update sparse info
      m_sparse_info.indices[iblk-nb_zero_blocks] = abs_blk_idx;
      m_sparse_info.offsets[iblk-nb_zero_blocks] = nb_nze_new;

      if (prev_row != row)
      {
        m_sparse_info.rows[row] = iblk-nb_zero_blocks;
      }

      chunk_start = (prev_was_empty) ? blk_off : chunk_start;
      chunk_size += blk_size;

      prev_was_empty = false;
      nb_nze_new += blk_size;
      nb_nzblocks_new += 1;

    }

    // move if necessary
    const bool move = (is_empty && !prev_was_empty) || (iblk == m_sinfo_local.nb_nzblocks-1);
    if (move)
    {
      std::memmove(m_win_data.data() + chunk_dest*float_size, 
                   m_win_data.data() + chunk_start*float_size, chunk_size*float_size);
    }

    if (is_empty)
    {
      if (!prev_was_empty)
      {
        // new destination for proceding chunk
        chunk_dest = blk_off;
        chunk_size = 0;
      }
      ++nb_zero_blocks;        
    }

    prev_row = row;

  }

  // get max number of blocks and elements
  MPI_Allreduce(&nb_nzblocks_new, &m_sinfo_local.nb_nzblocks_sym, 1, MPI_INT64_T, MPI_MAX, 
                m_ctx.get_comm());
  MPI_Allreduce(&nb_nze_new, &m_sinfo_local.nb_nze_sym, 1, MPI_INT64_T, MPI_MAX, m_ctx.get_comm());

  m_sinfo_local.nb_nzblocks = nb_nzblocks_new;
  m_sinfo_local.nb_nze = nb_nze_new;

  MPI_Allreduce(&nb_nzblocks_new, &m_sinfo_global.nb_nzblocks, 1, MPI_INT64_T, MPI_SUM, 
                m_ctx.get_comm()); 
  MPI_Allreduce(&nb_nze_new, &m_sinfo_global.nb_nze, 1, MPI_INT64_T, MPI_SUM, m_ctx.get_comm());

  m_sinfo_global.nb_nzblocks_sym = m_ctx.get_size() * m_sinfo_local.nb_nzblocks_sym;
  m_sinfo_global.nb_nze_sym = m_ctx.get_size() * m_sinfo_local.nb_nze_sym;

  // resize arrayss
  m_win_data.resize(m_sinfo_local.nb_nze_sym*float_size);

}

int64_t BaseTensor::get_max_block_size()
{
  int64_t size = 1;
  for (int i = 0; i < m_dim; ++i)
  {
    size *= *std::max_element(m_block_sizes[i].begin(), m_block_sizes[i].end());
  }
  return size;
}

double BaseTensor::get_occupation()
{
  return double(m_sinfo_global.nb_nzblocks)/double(m_sinfo_global.nb_blocks);
}

void BaseTensor::print_info()
{
  Log::print(m_logger, "Tensor Info on rank {}\n", m_ctx.get_rank());

  Log::print(m_logger, "Number of local blocks: {}/{}\n", 
             m_sinfo_local.nb_nzblocks, 
             m_sinfo_local.nb_blocks);

  Log::print(m_logger, "Number of local elements: {}/{}\n", 
             m_sinfo_local.nb_nze,
             m_sinfo_local.nb_elements);

  Log::print(m_logger, "Number of global blocks: {}/{}\n", 
             m_sinfo_global.nb_nzblocks, 
             m_sinfo_global.nb_blocks);

  Log::print(m_logger, "Number of local elements: {}/{}\n", 
             m_sinfo_global.nb_nze,
             m_sinfo_global.nb_elements);

  Log::print(m_logger, "Block sizes:\n");
  for (int idim = 0; idim < m_dim; ++idim)
  {
    Log::print(m_logger, "[{}]\n",  fmt::join(m_block_sizes[idim], ","));
  }

  Log::print(m_logger, "\n");
  Log::print(m_logger, "Sparsity Info:\n");

  Log::print(m_logger, "Occupation: {}\n", get_occupation());

  Log::print(m_logger, "Row offsets:\n");
  Log::print(m_logger, "[{}]\n", fmt::join(m_sparse_info.rows, ","));

  Log::print(m_logger, "Indices:\n");
  Log::print(m_logger, "[{}]\n", fmt::join(m_sparse_info.indices, ","));
  
  Log::print(m_logger, "Offsets:\n");
  Log::print(m_logger, "[{}]", fmt::join(m_sparse_info.offsets, ","));

  Log::print(m_logger, "\n");

}
  
#endif

} // namespace Shtensor