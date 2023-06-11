#ifndef SHTENSOR_CONTRACT_H
#define SHTENSOR_CONTRACT_H

#include "Tensor.h"

namespace Shtensor
{

template <typename T, int N1, int N2, int N3>
class Contract 
{
 public: 

  Contract(const std::string _expr, T _alpha, Tensor<T,N1>& _tensor_a, Tensor<T,N2>& _tensor_b, 
           T _beta, Tensor<T,N3>& _tensor_out)
    : m_tensor_a(_tensor_a)
    , m_tensor_b(_tensor_b)
    , m_tensor_c(_tensor_c)
    , m_alpha(_alpha)
    , m_beta(_beta)
  {
  }

  void perform(); 

 private:

  Tensor<T,N1>& m_tensor_a;
  Tensor<T,N2>& m_tensor_b;
  Tensor<T,N2>& m_tensor_c;

  T m_alpha;
  T m_beta;

};

template <typename T, int N1, int N2, int N3>
void Contract<T,N1,N2,N3>::perform()
{

}



}

#endif