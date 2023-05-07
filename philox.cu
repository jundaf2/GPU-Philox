#include "philox.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<float> gen_philox_rn(float range, int seed, int n){
  thrust::host_vector<float> h_vec(n);
  thrust::device_vector<float> d_vec(n);
  dim3 block(256);
  dim3 grid(((n/4) + block.x - 1) / block.x);
  philox_rnd<<<grid, block>>>(thrust::raw_pointer_cast(d_vec.data()), seed, n, range);
  cudaDeviceSynchronize();
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  py::array_t<float> result(n);
  auto result_buf = result.request();
  float *result_ptr = (float *) result_buf.ptr;
  std::copy(h_vec.begin(), h_vec.end(), result_ptr);
  return result;
}

PYBIND11_MODULE(cuPhilox, m) {
  m.doc() = "cuPhilox: a Philox PRNG used for fused kernel";
  m.def("gen_rn", &gen_philox_rn, "arguments: float range, int seed, int n");
}


