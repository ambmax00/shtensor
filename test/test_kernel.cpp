#include <mpi.h>
#include <Kernel.h>
#include <Logger.h>
#include "TestUtils.h"

#include <random>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#define PYSHARED(expr) \
  std::shared_ptr<PyObject> \

template <class T>
void fill_random(T* _pdata, int64_t _ssize)
{
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::generate(_pdata, _pdata+_ssize, [&](){ return dis(gen); });
}

template <class T, class SizeArray>
void get_statistics(T* _pdata, const SizeArray& _sizes, T& _rms)
{
  const int nb_elements = Shtensor::Utils::product(_sizes);
  const float square_sum = std::accumulate(_pdata, _pdata+nb_elements, T(), 
                                            [](auto _sum, auto _val)
                                            {
                                              return (_sum + std::pow(_val,2));
                                            });
  _rms = std::sqrt(square_sum/nb_elements);
}

#define CHECK(object) \
  if (!object)\
  {\
    PyErr_Print();\
    return false;\
  }

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

  fill_random<float>(data_a.data(), Shtensor::Utils::ssize(data_a));
  fill_random<float>(data_b.data(), Shtensor::Utils::ssize(data_b));
  //fill_random<float>(data_c.data(), Shtensor::Utils::ssize(data_c));

  Shtensor::Kernel<float> kernel(_expr, _sizes_a, _sizes_b, _sizes_c, 1.0, 0.0, 
                                 Shtensor::KernelMethod::LAPACK);

  fmt::print(kernel.get_info());

  kernel.call(data_a.data(), data_b.data(), data_c.data());

  float lapack_rms = 0;
  get_statistics(data_c.data(), _sizes_c, lapack_rms);

  fmt::print("RMS for LAPACK kernel is {}\n", lapack_rms);

  if (!einsum_python(_expr, 1.0, data_a.data(), _sizes_a, data_b.data(), _sizes_b,
                      0.0, data_c.data(), _sizes_c))
  {
    printf("Python failed\n");
    result += 1;
  }

  float python_rms = 0;
  get_statistics(data_c.data(), _sizes_c, python_rms);

  fmt::print("RMS for Python kernel is {}\n", python_rms);

  SHTENSOR_TEST_ALMOST_EQUAL(python_rms, lapack_rms, 1e-8, result);

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
    Py_Initialize();

    result += test_contract<2,2,2>("ik, kj -> ij", {5,6}, {6,8}, {5,8}, 10);

    result += test_contract<3,3,2>("ijk, kmj -> im", {5,6,8}, {8,4,6}, {5,4}, 10);

    Py_Finalize();

  }

  MPI_Finalize();

  return result;
}