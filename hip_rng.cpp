// This file implements a simple device-side PRNG using HIP. The PRNG uses
// xorwow.
#include <stdint.h>
#include <hip/hip_runtime.h>
#include "hip_rng.h"

// Increments the given value v and returns the new value, but skips 0.
__device__ uint64_t IncrementPastZero(uint64_t v) {
  v++;
  if (v != 0) return v;
  return 1;
}

__device__ uint32_t Generate32(RNGState *rng) {
  uint32_t *state = rng->s;
  uint32_t s, t;
  s = state[0];
  t = state[3];
  t ^= t >> 2;
  t ^= t << 1;
  state[3] = state[2];
  state[2] = state[1];
  state[1] = s;
  t ^= s;
  t ^= s << 4;
  state[0] = t;
  state[4] += 362437;
  return t + state[4];
}

__device__ double UniformDouble(RNGState *rng) {
  uint32_t a, b;
  uint64_t combined;
  a = Generate32(rng);
  b = Generate32(rng);
  combined = ((uint64_t) a) << 32;
  combined |= b;
  return ((double) combined) / ((double) ((uint64_t) -1));
}

__global__ void InitRNGState(uint64_t seed, int index, RNGState *s) {
  int i;
  for (i = 0; i < index; i++) {
    seed = IncrementPastZero(seed);
  }
  for (i = 0; i < 5; i++) {
    s->s[i] = seed;
    seed = IncrementPastZero(seed);
  }
  for (i = 0; i < 11; i++) {
    Generate32(s);
  }
}
