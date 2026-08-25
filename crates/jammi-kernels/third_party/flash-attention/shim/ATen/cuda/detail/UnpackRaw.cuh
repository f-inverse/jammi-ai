#pragma once
#include <tuple>
#include <cstdint>
namespace at { namespace cuda { namespace philox { __host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) { return std::make_tuple(arg.seed_, arg.offset_); } } } }
