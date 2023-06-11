#ifndef SHTENSOR_CONTRACT_INFO
#define SHTENSOR_CONTRACT_INFO 

#include <algorithm>
#include <map>
#include <vector>

#include "Logger.h"
#include "Utils.h"

namespace Shtensor 
{

struct ContractInfo
{
  std::vector<int> map_row;
  std::vector<int> map_col;
  std::vector<int> scatter_row;
  std::vector<int> scatter_col;

  void print(Log::Logger& _logger)
  {
    Log::print(_logger, "map_row: {}\n", fmt::join(map_row, ","));
    Log::print(_logger, "map_col: {}\n", fmt::join(map_col, ","));
    Log::print(_logger, "scatter row: {}\n", fmt::join(scatter_row, ","));
    Log::print(_logger, "scatter col: {}\n", fmt::join(scatter_col, ","));
  }
};

static inline std::vector<int> create_scatter_vector(const std::vector<int>& _map,   
                                                     const std::vector<int>& _sizes)
{
  const int scatter_size = std::accumulate(_map.begin(), _map.end(), 1, 
    [&_sizes](int _prod, int _val)
    {
      return _prod * _sizes[_val];
    });

  std::vector<int> scatter_vector(scatter_size);

  const auto strides = Utils::compute_strides(_sizes);

  for (int idx = 0; idx < scatter_size; ++idx)
  {
    int scatter_stride = 1;
    for (auto ax : _map)
    {
      // I += idx / scatter_stride mod size[ax] * stride[ax]
      scatter_vector[idx] += ((idx/scatter_stride)
                               %_sizes[ax])
                               *strides[ax];
    
      scatter_stride *= _sizes[ax];
    }
  }

  return scatter_vector;
}

template <class DimArrayIn1, class DimArrayIn2, class DimArrayOut>
bool compute_contract_info(const std::string& _expr, 
                           const DimArrayIn1& _dim_array_in1,
                           const DimArrayIn2& _dim_array_in2,
                           const DimArrayOut& _dim_array_out,
                           ContractInfo& _info_in1,
                           ContractInfo& _info_in2,
                           ContractInfo& _info_out,
                           std::string& _err_msg)
{

  if (_expr.empty())
  {
    _err_msg = "Equation is empty";
    return false;
  }

  enum TokenType
  {
    TOKEN_INDICES = 0,
    TOKEN_COMMA = 1,
    TOKEN_ARROW = 2
  };

  struct Token
  {
    TokenType type;
    int pos;
    std::string value;
  };

  // parse equation
  std::vector<Token> tokens;

  auto valid_index = [](char c) { return std::isalpha(c) || c == '_'; };

  for (int pos = 0; pos < Utils::ssize(_expr); ++pos)
  {
    const char c = _expr[pos];

    if (std::isspace(c))
    {
      continue;
    }

    if (valid_index(c))
    {
      std::string indices;

      int posoff = 0;
      char cc;

      while (valid_index((cc = _expr[pos+posoff])) && (pos+posoff < Utils::ssize(_expr)))
      {
        indices += cc;
        ++posoff;
      }

      tokens.push_back({TOKEN_INDICES,pos,indices});
      pos += posoff-1;

      continue;
    }

    if ((c == '-') && (pos+1 < Utils::ssize(_expr)) && (_expr[pos+1] == '>'))
    {
      tokens.push_back({TOKEN_ARROW,pos,"->"});
      pos++;
      continue;
    }

    if (c == ',')
    {
      tokens.push_back({TOKEN_COMMA,pos,","});
      continue;
    }

    _err_msg = fmt::format("Unknown character in equation at position {}: {}", pos, c);    
    return false;
  }

  // now analyze tokens
  // for (auto t : tokens)
  // {
  //   fmt::print("Token: {}, {}, {}\n", (int)t.type, t.pos, t.value);
  // }

  // valid snytax: (any indices) (comma) (any indices) (arrow) (any indices)
  int idx = 0;

  // map indices to integer axis 
  std::map<char,int> axis_map;

  #define EXPECTED(midx,mtype,mname) \
    if (midx >= Utils::ssize(tokens))\
    {\
      _err_msg = fmt::format("Unexpected end of expression, expected {}", #mname);\
      return false;\
    }\
    if (tokens[midx].type != mtype)\
    {\
      _err_msg = fmt::format("Expected {} at position {}", #mname, tokens[midx].pos);\
      return false;\
    }

  EXPECTED(idx,TOKEN_INDICES,index);

  auto map_to_idx = [&axis_map,axis=0](Token _token) mutable 
  { 
    std::vector<int> axes;
    for (auto c : _token.value)
    {
      if (axis_map.find(c) == axis_map.end()) axis_map[c] = axis++;
      axes.push_back(axis_map[c]);
    }
    return axes;
  };

  std::vector<int> indices_in1 = map_to_idx(tokens[idx]);

  ++idx;

  EXPECTED(idx,TOKEN_COMMA,",");

  ++idx;

  EXPECTED(idx,TOKEN_INDICES,index);

  std::vector<int> indices_in2 = map_to_idx(tokens[idx]);

  ++idx;

  EXPECTED(idx,TOKEN_ARROW,->);

  ++idx;

  EXPECTED(idx,TOKEN_INDICES,index);

  std::vector<int> indices_out = map_to_idx(tokens[idx]);

  ++idx;

  if (idx != Utils::ssize(tokens))
  {
    _err_msg = "Invalid syntax";
    return false;
  }

  #undef EXPECTED

  const std::set<int> set_in1(indices_in1.begin(), indices_in1.end());
  const std::set<int> set_in2(indices_in2.begin(), indices_in2.end());
  const std::set<int> set_out(indices_out.begin(), indices_out.end());

  // check if axes are not duplicate
  if ((set_in1.size() != indices_in1.size()) 
      || (set_in2.size() != indices_in2.size())
      || (set_out.size() != indices_out.size()))
  {
    _err_msg = "Duplicate tensor indices";
    return false;
  }

  // create maps
  std::set<int> ncon1;
  std::set<int> ncon2;
  std::set<int> con;

  std::set_intersection(set_in1.begin(),set_in1.end(),set_in2.begin(),set_in2.end(),
                        std::inserter(con,con.begin()));
  
  std::set_difference(set_in1.begin(),set_in1.end(),con.begin(),con.end(),
                      std::inserter(ncon1,ncon1.begin()));

  std::set_difference(set_in2.begin(),set_in2.end(),con.begin(),con.end(),
                      std::inserter(ncon2,ncon2.begin()));

  if (ncon1.size() == 0 || ncon2.size() == 0)
  {
    _err_msg = "Element-wise multiplication not supported";
    return false;
  }

  if (con.size() == 0)
  {
    _err_msg = "No indices found over which to contract";
    return false;
  }

  if (ncon1.size() + ncon2.size() != set_out.size())
  {
    _err_msg = "Dimension mismatch";
    return false;
  }

  // NEED TO MAP GLOBAL INDICES TO LOCAL INDICES FOR MAPS
  // CON IS NOT MAP

  auto to_local = [&axis_map](const auto& _all, const auto& _map)
  {
    std::map<int,int> global_to_local;
    for (int i = 0; i < Utils::ssize(_all); ++i)
    {
      global_to_local[_all[i]] = i;
    }
    
    std::vector<int> out;

    for (int s : _map)
    {
      out.push_back(global_to_local[s]);
    }

    return out;
  };

  _info_in1.map_row = to_local(indices_in1,ncon1);
  _info_in1.map_col = to_local(indices_in1,con);
  
  _info_in2.map_row = to_local(indices_in2,con);
  _info_in2.map_col = to_local(indices_in2,ncon2);

  _info_out.map_row = to_local(indices_out,ncon1);
  _info_out.map_col = to_local(indices_out,ncon2);

  auto check_sizes = [&_err_msg](const auto& _map1, const auto& _map2, 
                        const auto& _dims1, const auto& _dims2,
                        const std::string& _name1, const std::string& _name2)
  {
    for (int i = 0; i < Utils::ssize(_map1); ++i)
    {
      int iin = _dims1[_map1[i]];
      int iout = _dims2[_map2[i]];

      if (iin != iout)
      {
        _err_msg = fmt::format("Sizes for dimensions {} in {} and for dimension {} "
                               "in {} do not match.", _map1[i], _name1, _map2[i], _name2);
        return false;
      }
    }
    return true;
  };

  if (!check_sizes(_info_in1.map_row, _info_out.map_row, _dim_array_in1, _dim_array_out,
                   "tensor A", "tensor C"))
  {
    return false;
  }

  if (!check_sizes(_info_in2.map_col, _info_out.map_col, _dim_array_in2, _dim_array_out,
                   "tensor B", "tensor C"))
  {
    return false;
  }

  if (!check_sizes(_info_in1.map_col, _info_in2.map_row, _dim_array_in1, _dim_array_in2,
                   "tensor A", "tensor B"))
  {
    return false;
  }

  // create scatter vectors
  const std::vector<int> dims1(_dim_array_in1.begin(),_dim_array_in1.end());
  const std::vector<int> dims2(_dim_array_in2.begin(),_dim_array_in2.end());
  const std::vector<int> dims3(_dim_array_out.begin(),_dim_array_out.end());

  _info_in1.scatter_row = create_scatter_vector(_info_in1.map_row,dims1);
  _info_in1.scatter_col = create_scatter_vector(_info_in1.map_col,dims1);

  _info_in2.scatter_row = create_scatter_vector(_info_in2.map_row,dims2);
  _info_in2.scatter_col = create_scatter_vector(_info_in2.map_col,dims2);

  _info_out.scatter_row = create_scatter_vector(_info_out.map_row,dims3);
  _info_out.scatter_col = create_scatter_vector(_info_out.map_col,dims3);

  return true;

}

} // end Shtensor

#endif 