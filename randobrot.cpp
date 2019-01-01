#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

// The device number on which to perform the computation.
#define DEVICE_ID (0)

// This macro takes a hipError_t value and exits if it isn't equal to
// hipSuccess.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

// This defines the boundaries and locations of a "canvas" for drawing a
// complex-plane fractal.
typedef struct {
  // The width and height of the output image in pixels.
  int w;
  int h;
  // The boundaries of the image on the complex plane.
  double min_real;
  double min_imag;
  double max_real;
  double max_imag;
  // The distance between pixels in the real and imaginary axes.
  double delta_real;
  double delta_imag;
} FractalDimensions;

// This sets the minimum and/or maximum iterations for a fractal.
typedef struct {
  // This is the maximum number of iterations to run to see whether a point
  // escapes.
  uint32_t max_iterations;
  // The escape radius for the mandelbrot set. Must be at least 2.
  double escape_radius;
} IterationControl;

// This holds all data and settings for rendering a Mandelbrot set.
typedef struct {
  FractalDimensions dimensions;
  IterationControl iterations;
  // This is the data *on the device*.
  uint32_t *data;
} MandelbrotImage;

// This holds data required for coloring a pixel based on iterations.
typedef struct {
  // The minimum observed iterations in the canvas
  uint32_t min_iterations;
  // The maximum observed iterations in the canvas
  uint32_t max_iterations;
  // A scale to multiply the iterations by to obtain a linear color gradient.
  // TODO: remove when better coloring is possible.
  double linear_scale;
} IterationColorStats;

static void CleanupImage(MandelbrotImage *m) {
  if (m->data) hipFree(m->data);
  memset(m, 0, sizeof(*m));
}

// This returns the number of bytes needed to hold the 32-bit array of
// iterations for each output pixel.
static size_t GetBufferSize(MandelbrotImage *m) {
  return m->dimensions.w * m->dimensions.h * sizeof(uint32_t);
}

static void InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  exit(1);
}

// This uses a 2-D grid of threads. The x and y dimensions must cover the
// entire width of the output image.
__global__ void DrawMandelbrot(MandelbrotImage m) {
  int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if (col >= m.dimensions.w) return;
  if (row >= m.dimensions.h) return;
  int index = row * m.dimensions.w + col;
  uint32_t iteration;
  double tmp;
  double start_real = m.dimensions.delta_real * col + m.dimensions.min_real;
  double start_imag = m.dimensions.delta_imag * row + m.dimensions.min_imag;
  double current_real = start_real;
  double current_imag = start_imag;
  double radius = m.iterations.escape_radius;
  for (iteration = 0; iteration < m.iterations.max_iterations; iteration++) {
    if (((current_real * current_real) + (current_imag * current_imag)) >=
      radius) {
      break;
    }
    tmp = (current_real * current_real) - (current_imag * current_imag) +
      start_real;
    current_imag = 2 * current_real * current_imag + start_imag;
    current_real = tmp;
  }
  m.data[index] = iteration;
}

static void AllocateDeviceMemory(MandelbrotImage *m) {
  size_t bytes_needed = GetBufferSize(m);
  CheckHIPError(hipMalloc(&(m->data), bytes_needed));
}

// Update the delta_real and delta_imag values in the image dimensions.
static void UpdatePixelWidths(FractalDimensions *dims) {
  double tmp = dims->max_real - dims->min_real;
  dims->delta_real = tmp / ((double) dims->w);
  tmp = dims->max_imag - dims->min_imag;
  dims->delta_imag = tmp / ((double) dims->h);
}

// Copies image data from the device buffer to a host buffer.
static void CopyResults(MandelbrotImage *m, uint32_t *host_data) {
  size_t byte_count = GetBufferSize(m);
  CheckHIPError(hipMemcpy(host_data, m->data, byte_count,
    hipMemcpyDeviceToHost));
}

// Converts an iteration value to r, g, and b values.
static void IterationToColor(IterationColorStats *s, uint32_t iterations,
    uint8_t *r, uint8_t *g, uint8_t *b) {
  double v = (double) (iterations - s->min_iterations);
  int gray_value;
  v *= s->linear_scale;
  gray_value = (int) v;
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  *r = v;
  *g = v;
  *b = v;
}

// Converts host data to color image data for writing to a PPM image.
static void GetRGBImage(MandelbrotImage *m, uint32_t *host_data,
    uint8_t *color_data) {
  IterationColorStats stats;
  int x, y, w, h, index;
  double linear_scale = 0.0;
  uint32_t tmp;
  uint8_t r, g, b;
  uint32_t min_iterations = 0xffffffff;
  uint32_t max_iterations = 0;
  w = m->dimensions.w;
  h = m->dimensions.h;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      index = y * w + x;
      tmp = host_data[index];
      if (tmp < min_iterations) min_iterations = tmp;
      if (tmp > max_iterations) max_iterations = tmp;
    }
  }
  linear_scale = 255.0 / ((double) (max_iterations - min_iterations));
  stats.min_iterations = min_iterations;
  stats.max_iterations = max_iterations;
  stats.linear_scale = linear_scale;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      index = y * w + x;
      tmp = host_data[index];
      IterationToColor(&stats, tmp, &r, &g, &b);
      color_data[index * 3] = r;
      color_data[index * 3 + 1] = g;
      color_data[index * 3 + 2] = b;
    }
  }
}

// Saves a ppm-format file to the given filename. Returns 0 on error and
// nonzero otherwise.
static int SaveImage(MandelbrotImage *m, const char *filename) {
  FractalDimensions dims = m->dimensions;
  FILE *f = NULL;
  uint8_t *color_data = NULL;
  size_t color_data_size = 0;

  // First, copy the raw output buffer to the host.
  uint32_t *host_data = (uint32_t *) malloc(GetBufferSize(m));
  if (!host_data) {
    printf("Failed allocating buffer for host image.\n");
    return 0;
  }
  CopyResults(m, host_data);

  // Next, convert the raw buffer to RGB pixel data. We need 3 bytes per pixel
  // here.
  color_data_size = dims.w * dims.h * 3;
  color_data = (uint8_t *) malloc(color_data_size);
  if (!color_data) {
    printf("Failed allocating buffer for color image.\n");
    return 0;
  }
  GetRGBImage(m, host_data, color_data);
  free(host_data);
  host_data = NULL;

  // Next, create the output file and write the data to it.
  f = fopen(filename, "wb");
  if (!f) {
    printf("Failed opening output file: %s\n", strerror(errno));
    free(color_data);
    return 0;
  }
  // Include the parameters for the set as a "comment" in the image.
  if (fprintf(f, "P6\n# Bounds: (%f, %fi, %f, %fi). Max iters: %u\n",
    dims.min_real, dims.min_imag, dims.max_real, dims.max_imag,
    (unsigned) m->iterations.max_iterations) <= 0) {
    printf("Failed writing Mandelbrot metadata: %s\n", strerror(errno));
    fclose(f);
    free(color_data);
    return 0;
  }
  if (fprintf(f, "%d %d\n255\n", dims.w, dims.h) <= 0) {
    printf("Failed writing ppm header: %s\n", strerror(errno));
    fclose(f);
    free(color_data);
    return 0;
  }

  if (!fwrite(color_data, color_data_size, 1, f)) {
    printf("Failed writing pixel data: %s\n", strerror(errno));
    fclose(f);
    free(color_data);
    return 0;
  }
  fclose(f);
  free(color_data);
  return 1;
}

// Initializes the Mandelbrot image with some basic defaults.
static void InitializeImage(MandelbrotImage *m, int w, int h) {
  if ((w <= 0) || (h <= 0)) {
    printf("Bad image width or height.\n");
    exit(1);
  }
  m->dimensions.w = w;
  m->dimensions.h = h;
  m->dimensions.min_real = -2;
  m->dimensions.min_imag = -2;
  m->dimensions.max_real = 2;
  m->dimensions.max_imag = 2;
  m->iterations.max_iterations = 100;
  m->iterations.escape_radius = 2;
  UpdatePixelWidths(&(m->dimensions));
  AllocateDeviceMemory(m);
}

// Sets the HIP device and prints out some device info.
static void SetupDevice(void) {
  printf("Setting device...\n");
  hipDeviceProp_t props;
  CheckHIPError(hipSetDevice(DEVICE_ID));
  CheckHIPError(hipGetDeviceProperties(&props, DEVICE_ID));
  printf("Running on device: %s\n", props.name);
  printf("Using AMD GPU architecture %d. Has %d multiprocessors, "
    "supporting %d threads each.\n", props.gcnArch, props.multiProcessorCount,
    props.maxThreadsPerMultiProcessor);
}

int main(int argc, char **argv) {
  MandelbrotImage m;
  SetupDevice();
  InitializeImage(&m, 1000, 1000);
  // Blocks are 16x16 threads
  dim3 block_dim(16, 16);
  dim3 grid_dim((1000 / 16) + 1, (1000 / 16) + 1);
  hipLaunchKernelGGL(DrawMandelbrot, grid_dim, block_dim, 0, 0, m);
  CheckHIPError(hipDeviceSynchronize());
  if (!SaveImage(&m, "output.ppm")) {
    printf("Failed saving image.\n");
  } else {
    printf("Image saved OK.\n");
  }
  CleanupImage(&m);
  return 0;
}
