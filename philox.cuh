#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <array>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>




__global__ void philox_rnd (float* buf, unsigned long long seed, unsigned n, float range) {
	constexpr unsigned long kPhilox10A = 0x9E3779B9;
	constexpr unsigned long kPhilox10B = 0xBB67AE85;
	constexpr unsigned long kPhiloxSA = 0xD2511F53;
	constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
	constexpr float M_RAN_INVM32 = 2.3283064e-10f;

	uint2 key = reinterpret_cast<const uint2&>(seed);
  uint4 counter = make_uint4(0, 0, 0, 0);
  unsigned long long subsequence = 0;
  unsigned long long offset = (blockIdx.x*blockDim.x + threadIdx.x);
	counter.x += (unsigned int) (offset);
	counter.y += (unsigned int) (offset >> 32);
  counter.z += (unsigned int)(subsequence);
  counter.w += (unsigned int)(subsequence >> 32);
	
	for(int round = 0; round < 7; round++){
		unsigned int hi0 = __umulhi(kPhiloxSA, counter.x);
		unsigned int hi1 = __umulhi(kPhiloxSB, counter.z);
		unsigned int lo0 = kPhiloxSA * counter.x;
		unsigned int lo1 = kPhiloxSB * counter.z;
		counter = {hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0};
		key.x += (kPhilox10A);
    key.y += (kPhilox10B);
	}
  float4 ret = make_float4(counter.x * M_RAN_INVM32 * range, counter.y * M_RAN_INVM32 * range, counter.z * M_RAN_INVM32 * range, counter.w * M_RAN_INVM32 * range);
  if(offset*4<n) reinterpret_cast<float4*>(buf)[offset] = ret;
}