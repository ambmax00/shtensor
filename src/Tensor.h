#ifndef SHTENSOR_TENSOR_H
#define SHTENSOR_TENSOR_H

#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "BlockSpan.h"
#include "Context.h"
#include "MemoryPool.h"

namespace Shtensor 
{

template <int N>
using VArray = std::array<std::vector<int>,N>;

template <typename T, int N>
class Tensor 
{

  static_assert(N >= 1, "Tensor has to have dimension >= 1");

 public: 
  
  Tensor(const Context& _ctx, const VArray<N>& _block_sizes, MemoryPool& _mempool) noexcept
    : m_ctx(_ctx) 
    , m_block_sizes(_block_sizes) 
    , m_pool(_mempool)
    , m_grid_dims({})
    , m_p_cart(nullptr)
    , m_distributions({})
    , m_p_data(nullptr)
    , m_sparse_info()
    , m_nb_elements_local(0)
    , m_nb_blocks_local(0)
    , m_nb_elements_global(0)
    , m_nb_blocks_global(0)
    , m_nb_nze_local(0)
    , m_nb_nzblocks_local(0)
  {
    // create cartesian grid
    std::fill(m_grid_dims.begin(), m_grid_dims.end(), 0);

    MPI_Dims_create(m_ctx.get_size(), N, m_grid_dims.data());

    m_p_cart.reset(new MPI_Comm(MPI_COMM_NULL), Context::s_comm_deleter);

    std::array<int,N> periods;
    std::fill(periods.begin(), periods.end(), 1);

    MPI_Cart_create(m_ctx.get_comm(), N, m_grid_dims.data(), periods.data(), 0, m_p_cart.get());

    MPI_Cart_coords(*m_p_cart, m_ctx.get_rank(), N, m_coords.data());

    // create distributions
    for (int i = 0; i < N; ++i)
    {
      m_distributions[i] = Utils::compute_default_dist(m_block_sizes[i].size(), 
                                                       m_grid_dims[i], m_block_sizes[i]);
    }

    m_nb_blocks_global = std::accumulate(_block_sizes.begin(), _block_sizes.end(), 1.0,
                          [](int64_t _prod, const auto& _sizes)
                          {
                            return _prod *= Utils::ssize(_sizes);
                          });

    std::generate(m_block_dims.begin(), m_block_dims.end(), 
                  [i=0,this]() mutable { return Utils::ssize(m_block_sizes[i++]); });

    m_block_strides = Utils::compute_strides(m_block_dims);

    // get local indices
    for (int idim = 0; idim < N; ++idim)
    {
      for (int iblk = 0; iblk < m_block_dims[idim]; ++iblk)
      {
        if (m_distributions[idim][iblk] == m_coords[idim])
        {
          m_block_idx_local[idim].push_back(iblk);
        }
      }
    }

    // get total number of local blocks
    
  }

  ~Tensor()
  {
  }

  VArray<N> get_local_indices() const 
  {
    //VArray<N> out;




  }

  std::size_t get_dim_size(int i)
  {
    //if (i < N || i >= N) return 0;

    //return std::accumulate(m_block_sizes[i].begin(), m_block_sizes[i].end(), std::size_t(0));
  }

  void reserve(const VArray<N>& _block_idx)
  {
    bool equalSize = std::equal(_block_idx.begin()+1, _block_idx.end(), _block_idx.begin(), 
      [](const auto& _idx0, const auto& _idx1)
      {
        return _idx0.size() == _idx1.size();
      });

    if (!equalSize)
    {
      throw std::runtime_error("reserve: Indices do not have the same size in each dimension");
    }

    const int64_t nb_rows = Utils::ssize(m_block_sizes[0]);

    const int64_t nb_blocks = Utils::ssize(_block_idx[0]);

    // count slices in each row and the number of total elements in that row
    std::vector<int64_t> nb_slices_row(nb_rows, 0);
    std::vector<int64_t> nb_elements_row(nb_rows, 0);

    for (int64_t iblk = 0; iblk < nb_blocks; ++iblk)
    {
      const int64_t irow = _block_idx[0][iblk];
      nb_slices_row[irow]++;

      int block_size = 1;

      for (auto idim = 0ul; idim < N; ++idim)
      {
        block_size *= m_block_sizes[idim][_block_idx[idim][iblk]];
      }

      nb_elements_row[irow] += block_size;
    }

    // reserve space for sparsity info
    const int64_t sparse_data_size = (nb_rows+2*nb_blocks)*SSIZEOF(int64_t);

    m_sparse_info.p_raw.reset(m_pool.allocate<uint8_t>(sparse_data_size),
                              get_pool_deleter<uint8_t>());

    m_sparse_info.row_idx = Span<int64_t>(
      reinterpret_cast<int64_t*>(m_sparse_info.p_raw.get()), nb_rows);

    m_sparse_info.slice_idx = Span<int64_t>(m_sparse_info.row_idx.begin() + nb_rows, nb_blocks);

    m_sparse_info.slice_offset = Span<int64_t>(
      reinterpret_cast<int64_t*>(m_sparse_info.slice_idx.begin() + nb_blocks), nb_blocks);
  
    std::copy(nb_slices_row.begin(), nb_slices_row.end(), m_sparse_info.row_idx.begin());

    // create rolled index array
    constexpr int nb_rolled_dims = N-1;
    m_sparse_info.strides[0] = 1;
    
    for (int idim = 1; idim < nb_rolled_dims; ++idim)
    {
      m_sparse_info.strides[idim] = m_sparse_info.strides[idim-1]
                                    *Utils::ssize(m_block_sizes[idim]);
    }

    std::vector<int64_t> rolled_idx(nb_blocks, 0);

    for (int iblk = 0; iblk < nb_blocks; ++iblk)
    {
      for (int idim = 0; idim < nb_rolled_dims; ++idim)
      {
        rolled_idx[iblk] += _block_idx[idim+1][iblk]*m_sparse_info.strides[idim];
      } 
    }

    // sort indices
    std::vector<int64_t> perm(nb_blocks);
    std::iota(perm.begin(), perm.end(), 0);

    std::sort(perm.begin(), perm.end(), 
      [&_block_idx,&rolled_idx](int64_t val0, int64_t val1)
      {
        return (_block_idx[val0] == _block_idx[val1]) 
                ? (rolled_idx[val0] < rolled_idx[val1])
                : (_block_idx[0][val0] < _block_idx[0][val1]);
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
      m_sparse_info.row_idx[irow] = offset;
      std::copy(sorted_rolled_idx.begin()+offset, 
                sorted_rolled_idx.begin()+offset+nb_slices_row[irow], 
                m_sparse_info.slice_idx.begin()+offset);

      offset += nb_slices_row[irow];
    }

    offset = 0;
    int64_t iblk = 0;

    // go through each block and record offset
    for (int64_t irow = 0; irow < nb_rows; ++irow)
    {
      for (int64_t islice = 0; islice < nb_slices_row[irow]; ++islice)
      {
        std::array<int64_t,N-1> indices = {};
        Utils::unroll_index(m_sparse_info.strides, m_sparse_info.slice_idx[iblk], indices);
        
        int blk_size = m_block_sizes[0][irow];
        
        for (int idim = 1; idim < N; ++idim)
        {
          blk_size *= m_block_sizes[idim][indices[idim-1]];
        }

        m_sparse_info.slice_offset[iblk] = offset;

        offset += blk_size;
        iblk++;
      }
    }

    // count number of total elements
    m_nb_nze_local = std::accumulate(nb_elements_row.begin(), nb_elements_row.end(), 0);

    // allocate full space and set to zero
    m_p_data.reset(m_pool.allocate<T>(m_nb_nze_local), get_pool_deleter<T>());
    std::fill(m_p_data.get(), m_p_data.get()+m_nb_nze_local, T());
    
  }

  
  void print_info()
  {
    printf("Tensor Info one rank %d\n", m_ctx.get_rank());

    printf("Block sizes:\n");
    for (int idim = 0; idim < N; ++idim)
    {
      for (int iblk = 0l; iblk < Utils::ssize(m_block_sizes[idim]); ++iblk)
      {
        printf("%d ", m_block_sizes[idim][iblk]);
      }
      printf("\n");
    }

    printf("\n");
    printf("Sparsity Info:\n");

    printf("Number of local nze: %ld\n", m_nb_nze_local);
    printf("Occupation: %f\n", 0.f);

    printf("Row offsets:\n");
    
    for (auto offset : m_sparse_info.row_idx)
    {
      printf("%ld ", offset);
    }

    printf("\nStrides:\n");

    for (auto s : m_sparse_info.strides)
    {
      printf("%ld ", s);
    }

    printf("\nIndices:\n");

    for (auto idx : m_sparse_info.slice_idx)
    {
      printf("%ld ", idx);
    }

    printf("\nOffsets:\n");

    for (auto off : m_sparse_info.slice_offset)
    {
      printf("%ld ", off);
    }

    printf("\n\n");

  }
  
  template <typename D>
  std::function<void(D*)> get_pool_deleter()
  {
    return [memPool=m_pool](D* _ptr) mutable { memPool.free(_ptr); };
  }

  const Span<int64_t>& get_row_idx() const
  {
    return m_sparse_info.row_idx;
  }

  const Span<int64_t>& get_slice_idx() const 
  {
    return m_sparse_info.slice_idx;
  }

  double get_occupation()
  {
    // NYI
    return 0.0;
  }

 private: 

  const Context m_ctx;

  const VArray<N> m_block_sizes;

  MemoryPool m_pool;

  std::array<int,N> m_grid_dims;

  std::shared_ptr<MPI_Comm> m_p_cart;

  VArray<N> m_distributions;

  std::shared_ptr<T> m_p_data;

  struct SparseInfo
  {
    std::shared_ptr<uint8_t> p_raw;
    Span<int64_t> row_idx;
    Span<int64_t> slice_idx;
    Span<int64_t> slice_offset;
    std::array<int64_t,N-1> strides;
  };
  
  SparseInfo m_sparse_info;

  int64_t m_nb_elements_local;

  int64_t m_nb_blocks_local;

  int64_t m_nb_elements_global;

  int64_t m_nb_blocks_global;

  int64_t m_nb_nze_local;

  int64_t m_nb_nzblocks_local;

  std::array<int64_t,N> m_block_dims;

  std::array<int64_t,N> m_block_strides;

  std::array<int,N> m_coords;

  VArray<N> m_block_idx_local;

};

/*template <class T, int N>
class BlockView
{
 public:

  class BlockIterator 
  {
    public:
      using iterator_category = std::random_access_iterator_tag;
      using difference_type   = std::ptrdiff_t;
      using value_type        = BlockSpan<T,N>;
      using pointer           = BlockSpan<T,N>*;  // or also value_type*
      using reference         = BlockSpan<T,N>&;  // or also value_type&

      BlockIterator(Tensor<T,N>& _tensor, std::size_t _idx)
        : m_tensor(_tensor)
        , m_block_idx(_idx)
        , m_indices()
      {

      }

    private:

      std::size_t m_block_idx;

      std::array<int,N> m_indices;
  };

 private:

  Tensor<N,T> m_tensor;

};*/

}

#endif