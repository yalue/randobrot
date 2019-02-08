// This defines a basic device-side library for generating pseudorandom
// numbers in HIP.
#ifndef HIP_RNG_H
#define HIP_RNG_H
#include <stdint.h>
#include <hip/hip_runtime.h>

// Holds the state for the Xorwow PRNG.
typedef struct {
  uint32_t s[5];
} RNGState;

// Initializes an RNG state in device code.
__device__ void InitRNGState(uint64_t seed, int index, RNGState *s);

// Generates a double between [0, 1] with a uniform distribution.
__device__ double UniformDouble(RNGState *rng);

// Generates a 32-bit random number from the RNG state.
__device__ uint32_t Generate32(RNGState *rng);

#endif  // HIP_RNG_H

