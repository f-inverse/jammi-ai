#pragma once
#include <cstdint>
namespace at { struct PhiloxCudaState { PhiloxCudaState() = default; PhiloxCudaState(uint64_t seed, uint64_t offset) : seed_(seed), offset_(offset) {} uint64_t seed_ = 0; uint64_t offset_ = 0; }; struct Generator {}; }
