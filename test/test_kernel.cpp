#include <mpi.h>
#include <Kernel.h>
#include <Logger.h>
#include <Timer.h>
#include "TestUtils.h"

#include <random>

//#define WITH_PYTHON
#ifdef WITH_PYTHON
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #include <Python.h>
  #include <numpy/arrayobject.h>

  #define CHECK(object) \
  if (!object)\
  {\
    PyErr_Print();\
    return false;\
  }
#endif

template <class Iterator>
void fill_random(Iterator _pbegin, Iterator _pend)
{
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::generate(_pbegin, _pend, [&](){ return dis(gen); });
}

template <class Iterator>
void get_statistics(Iterator _pbegin, Iterator _pend, typename Iterator::value_type& _rms)
{
  const float square_sum = std::accumulate(_pbegin, _pend, typename Iterator::value_type(), 
                                            [](auto _sum, auto _val)
                                            {
                                              return (_sum + std::pow(_val,2));
                                            });
  _rms = std::sqrt(square_sum/int64_t(_pend-_pbegin));
}

#ifdef WITH_PYTHON
template <class SizeArrayA, class SizeArrayB, class SizeArrayC>
bool einsum_python(const std::string& _expr, 
                   float _alpha, 
                   float* _a, 
                   const SizeArrayA& _sizes_a,
                   float* _b,
                   const SizeArrayB& _sizes_b,
                   float _beta,
                   float* _c,
                   const SizeArrayC _sizes_c
                   )
{
  PyObject* np_module = PyImport_ImportModule("numpy");

  CHECK(np_module);

  import_array();
  
  PyObject* np_einsum = PyObject_GetAttrString(np_module, "einsum");

  CHECK(np_einsum);

  std::vector<npy_intp> lsizes_a(_sizes_a.begin(), _sizes_a.end());
  std::vector<npy_intp> lsizes_b(_sizes_b.begin(), _sizes_b.end());
  std::vector<npy_intp> lsizes_c(_sizes_c.begin(), _sizes_c.end());

  // Convert input data to NumPy arrays
  PyObject* a_py = PyArray_New(&PyArray_Type, lsizes_a.size(), lsizes_a.data(), NPY_FLOAT32,
                               nullptr, _a, 0, NPY_ARRAY_FARRAY, nullptr);
  CHECK(a_py);

  PyObject* b_py = PyArray_New(&PyArray_Type, lsizes_b.size(), lsizes_b.data(), NPY_FLOAT32,
                               nullptr, _b, 0, NPY_ARRAY_FARRAY, nullptr);
  CHECK(b_py);

  PyObject* c_py = PyArray_New(&PyArray_Type, lsizes_c.size(), lsizes_c.data(), NPY_FLOAT32,
                               nullptr, _c, 0, NPY_ARRAY_FARRAY, nullptr);
  CHECK(c_py);

  // Create arguments tuple for einsum function
  PyObject* einsum_args = PyTuple_New(3);
  CHECK(einsum_args);

  PyTuple_SetItem(einsum_args, 0, PyUnicode_FromString(_expr.c_str()));
  PyTuple_SetItem(einsum_args, 1, a_py);
  PyTuple_SetItem(einsum_args, 2, b_py);

  // Create keyword arguments dictionary for einsum function
  PyObject* einsum_kwargs = PyDict_New();
  CHECK(einsum_kwargs);

  PyDict_SetItemString(einsum_kwargs, "order", PyUnicode_FromString("F"));

  // Call the einsum function
  PyObject* result = PyObject_Call(np_einsum, einsum_args, einsum_kwargs);
  CHECK(result);

  // Copy the result to the output tensor
  const int64_t nb_elements_c = Shtensor::Utils::product(_sizes_c);
  memcpy(_c, PyArray_DATA(reinterpret_cast<PyArrayObject*>(result)), nb_elements_c*sizeof(float));

  // Cleanup
  Py_DECREF(result);
  Py_DECREF(c_py);
  Py_DECREF(einsum_kwargs);
  Py_DECREF(einsum_args);
  Py_DECREF(np_einsum);
  Py_DECREF(np_module);

  return true;

}
#endif

template <int N, int M, int K>
int test_contract(const std::string _expr, 
                  const std::array<int,N>& _sizes_a,
                  const std::array<int,M>& _sizes_b,
                  const std::array<int,K>& _sizes_c,
                  int _nb_cons)
{
  int result = 0;

  auto logger = Shtensor::Log::create("test_contract");

  const int nb_elements_a = Shtensor::Utils::product(_sizes_a);
  const int nb_elements_b = Shtensor::Utils::product(_sizes_b);
  const int nb_elements_c = Shtensor::Utils::product(_sizes_c);

  std::vector<float> data_a(_nb_cons*nb_elements_a,0);
  std::vector<float> data_b(_nb_cons*nb_elements_b,0);
  std::vector<float> data_c(_nb_cons*nb_elements_c,0);
  std::vector<float> result_c(_nb_cons*nb_elements_c,0);

  fill_random(data_a.begin(), data_a.end());
  fill_random(data_b.begin(), data_b.end());
  fill_random(data_c.begin(), data_c.end());

  Shtensor::Kernel<float> kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 1.0, 0.0, 
                                 Shtensor::KernelMethod::LAPACK);

  fmt::print(kernel.get_info());

  std::copy(data_c.begin(), data_c.end(), result_c.begin());
  kernel.call(data_a.data(), data_b.data(), result_c.data(), _nb_cons);

  float lapack_rms = 0;
  get_statistics(result_c.begin(), result_c.end(), lapack_rms);

  // fmt::print("C: {}", fmt::join(result_c.begin(), result_c.end(), ","));

  Shtensor::Kernel<float> xmm_kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 1.0, 0.0, 
                                     Shtensor::KernelMethod::XMM);

  std::copy(data_c.begin(), data_c.end(), result_c.begin());
  xmm_kernel.call(data_a.data(), data_b.data(), result_c.data(), _nb_cons);

  float xmm_rms = 0;
  get_statistics(result_c.begin(), result_c.end(), xmm_rms);

  // fmt::print("A: [{}]\n", fmt::join(data_a.begin(), data_a.end(), ","));
  // fmt::print("B: [{}]\n", fmt::join(data_b.begin(), data_b.end(), ","));
  // fmt::print("C: [{}]\n", fmt::join(result_c.begin(), result_c.end(), ","));

#ifdef WITH_PYTHON
  std::copy(data_c.begin(), data_c.end(), result_c.begin());
  if (!einsum_python(_expr, 1.0, data_a.data(), _sizes_a, data_b.data(), _sizes_b,
                      0.0, data_c.data(), _sizes_c))
  {
    printf("Python failed\n");
    result += 1;
  }

  float python_rms = 0;
  get_statistics(result_c.begin(), result_c.end(), python_rms);

  // fmt::print("RMS for Python kernel is {}\n", python_rms);

  SHTENSOR_TEST_ALMOST_EQUAL(python_rms, lapack_rms, 1e-6, result);
#endif

  SHTENSOR_TEST_ALMOST_EQUAL(lapack_rms, xmm_rms, 1e-6, result);

  return result;
}

template <int N, int M, int K>
int test_timings(const std::string _expr, 
                 const std::array<int,N>& _sizes_a,
                 const std::array<int,M>& _sizes_b,
                 const std::array<int,K>& _sizes_c,
                 int _nb_cons)
{
  int result = 0;

  auto logger = Shtensor::Log::create("test_timings");

  const int nb_elements_a = Shtensor::Utils::product(_sizes_a);
  const int nb_elements_b = Shtensor::Utils::product(_sizes_b);
  const int nb_elements_c = Shtensor::Utils::product(_sizes_c);

  std::vector<float> data_a(_nb_cons*nb_elements_a, _nb_cons);
  std::vector<float> data_b(_nb_cons*nb_elements_b, _nb_cons);
  std::vector<float> data_c(_nb_cons*nb_elements_c, _nb_cons);
  std::vector<float> result_c(_nb_cons*nb_elements_c, _nb_cons);

  fill_random(data_a.begin(), data_a.end());
  fill_random(data_b.begin(), data_b.end());
  fill_random(data_c.begin(), data_c.end());

  Shtensor::Kernel<float> kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 1.0, 0.0, 
                                 Shtensor::KernelMethod::LAPACK);

  fmt::print(kernel.get_info());

  std::copy(data_c.begin(), data_c.end(), result_c.begin());

  Shtensor::Timer lapack_timer;
  kernel.call(data_a.data(), data_b.data(), result_c.data(), _nb_cons);
  const double lapack_time = lapack_timer.elapsed();

  Shtensor::Kernel<float> xmm_kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 1.0, 0.0, 
                                     Shtensor::KernelMethod::XMM);

  std::copy(data_c.begin(), data_c.end(), result_c.begin());

  Shtensor::Timer xmm_timer;
  xmm_kernel.call(data_a.data(), data_b.data(), result_c.data(), _nb_cons);
  const double xmm_time = xmm_timer.elapsed();

  fmt::print("Times: {}s, {}s", lapack_time/_nb_cons, xmm_time/_nb_cons);

  return result;
}


template <int N, int M, int K>
int test_kernel(const std::string _expr, 
                const std::array<int,N>& _sizes_a,
                const std::array<int,M>& _sizes_b,
                const std::array<int,K>& _sizes_c)
{
  int result = 0;

  auto logger = Shtensor::Log::create("test_xmm");

  const int nb_elements_a = Shtensor::Utils::product(_sizes_a);
  const int nb_elements_b = Shtensor::Utils::product(_sizes_b);
  const int nb_elements_c = Shtensor::Utils::product(_sizes_c);

  std::vector<float> data_a(nb_elements_a,0);
  std::vector<float> data_b(nb_elements_b,0);
  std::vector<float> data_c(nb_elements_c,0);

  fill_random(data_a.begin(), data_a.end());
  fill_random(data_b.begin(), data_b.end());
  fill_random(data_c.begin(), data_c.end());

  const float sum_c = std::accumulate(data_c.begin(), data_c.end(), 0.0);

  // fmt::print("A: [{}]\n", fmt::join(data_a.begin(), data_a.end(), ","));
  // fmt::print("B: [{}]\n", fmt::join(data_b.begin(), data_b.end(), ","));
  // fmt::print("C: [{}]\n", fmt::join(data_c.begin(), data_c.end(), ","));

  const float beta = 2.0;
  Shtensor::Kernel<float> kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 0.0, beta, 
                                 Shtensor::KernelMethod::XMM);

  kernel.call(data_a.data(), data_b.data(), data_c.data(), 1);

  // fmt::print("C: [{}]\n", fmt::join(data_c.begin(), data_c.end(), ","));

  const float sum_c_after = std::accumulate(data_c.begin(), data_c.end(), 0.0);

  SHTENSOR_TEST_ALMOST_EQUAL(beta*sum_c, sum_c_after, 1e-6, result);

  return result;

}



int main(int argc, char** argv)
{
  auto logger = Shtensor::Log::create("test");

  MPI_Init(&argc,&argv);
  int rank = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int result = 0;

  if (rank == 0)
  {
    #ifdef WITH_PYTHON
    Py_Initialize();
    #endif

    result += test_contract<2,2,2>("ik, kj -> ij", {8,5}, {5,6}, {8,6}, 20);

    result += test_contract<3,3,2>("jik, kmj -> im", {6,5,8}, {8,4,6}, {5,4}, 20);

    result += test_kernel<4,3,3>("ijml, lmk -> jki", {5,6,8,9}, {9,8,7}, {6,7,5});
    
    // very very small matrices
     result += test_kernel<2,2,2>("ij, jk -> ik", {2,2}, {2,2}, {2,2});

    // very small matrices
    result += test_kernel<2,2,2>("ij, jk -> ik", {5,5}, {5,5}, {5,5});

    // Max size matrix that fits in one register
    result += test_kernel<2,2,2>("ij, jk -> ik", {8,8}, {8,8}, {8,8});

    // Max size matrix that fits in one register
    result += test_kernel<2,2,2>("ij, jk -> ik", {10,10}, {10,10}, {10,10});

    result += test_timings<3,3,2>("ikl, ljk -> ij", {8,8,8}, {8,8,8}, {8,8}, 100000);

    result += test_timings<3,3,2>("kil, ljk -> ij", {8,8,8}, {8,8,8}, {8,8}, 100000);

    result += test_timings<3,3,2>("ikl, ljk -> ji", {5,5,5}, {5,5,5}, {5,5}, 100000);

    result += test_timings<2,2,2>("ik, kj -> ij", {8,8}, {8,8}, {8,8}, 100000);

    result += test_timings<2,2,2>("ik, kj -> ij", {16,16}, {16,16}, {16,16}, 100000);
    
    #ifdef WITH_PYTHON
    Py_Finalize();
    #endif

  }

  MPI_Finalize();

  return result;
}