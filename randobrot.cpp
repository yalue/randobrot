#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

// The device number on which to perform the computation.
#define DEVICE_ID (0)

// The number of GPU threads to use when searching for interesting locations.
#define SEARCH_THREAD_COUNT (2048)

// The number of times each thread retries searching for a good location.
#define RETRIES_PER_THREAD (512)

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
  // This is the minimum number of iterations to run when searching for an
  // interesting starting point. Unused when rendering a full image.
  uint32_t min_iterations;
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

// This contains parameters and outputs used when searching for interesting
// locations.
typedef struct {
  // A list of RNG states for the search. There must be one such state per
  // search thread.
  hiprandState_t *rng_states;
  // A list of real locations, 1 per thread, containing the interesting point.
  // This will be less than -2 if the associated thread did not find an
  // interesting point.
  double *real_locations;
  // Like real_locations, except holds the imaginary location for corresponding
  // threads.
  double *imag_locations;
  // This holds the iterations needed for the interesting point in each
  // corresponding thread. This will be set to 0 or max_iterations if no
  // interesting point was found by the thread.
  uint32_t *iterations_needed;
  IterationControl iterations;
} RandomSearchData;

// This holds device-side data for coloring an image.
typedef struct {
  // The number of entries in the histogram
  uint32_t max_iterations;
  // The histogram mapping iteration -> # of pixels with that iteration.
  uint64_t *histogram;
  // The number of entries in the raw data buffer
  uint64_t pixel_count;
  // The original raw image data
  uint32_t *data;
  // The RGB buffer to be filled
  uint8_t *rgb_data;
} HistogramColorData;

static void InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  exit(1);
}

// Uses the current time in nanoseconds to get an RNG seed.
static uint64_t GetRNGSeed(void) {
  uint64_t to_return = 0;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  to_return = t.tv_nsec;
  to_return += t.tv_sec * 1e9;
  return to_return;
}

static void CleanupImage(MandelbrotImage *m) {
  if (m->data) hipFree(m->data);
  memset(m, 0, sizeof(*m));
}

// This returns the number of bytes needed to hold the 32-bit array of
// iterations for each output pixel.
static size_t GetBufferSize(MandelbrotImage *m) {
  return m->dimensions.w * m->dimensions.h * sizeof(uint32_t);
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
  m->iterations.min_iterations = 0;
  m->iterations.max_iterations = 100;
  m->iterations.escape_radius = 2;
  UpdatePixelWidths(&(m->dimensions));
  AllocateDeviceMemory(m);
}

__global__ void InitializeRNG(uint64_t seed, hiprandState_t *states) {
  int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (index > SEARCH_THREAD_COUNT) return;
  hiprand_init(seed, index, 0, states + index);
}

// Initializes the GPU memory and RNG states for the random search.
static void InitializeRandomSearchData(RandomSearchData *d,
    IterationControl *c) {
  int count = SEARCH_THREAD_COUNT;
  int block_count = count / 256;
  if ((count % 256) != 0) block_count++;
  CheckHIPError(hipMalloc(&(d->rng_states), count * sizeof(hiprandState_t)));
  CheckHIPError(hipMalloc(&(d->real_locations), count * sizeof(double)));
  CheckHIPError(hipMemset(d->real_locations, 0, count * sizeof(double)));
  CheckHIPError(hipMalloc(&(d->imag_locations), count * sizeof(double)));
  CheckHIPError(hipMemset(d->imag_locations, 0, count * sizeof(double)));
  CheckHIPError(hipMalloc(&(d->iterations_needed), count * sizeof(uint32_t)));
  CheckHIPError(hipMemset(d->iterations_needed, 0, count * sizeof(uint32_t)));
  d->iterations = *c;
  hipLaunchKernelGGL(InitializeRNG, block_count, 256, 0, 0, GetRNGSeed(),
    d->rng_states);
  CheckHIPError(hipDeviceSynchronize());
}

// Frees all of the memory in the RandomSearchData struct.
static void CleanupRandomSearchData(RandomSearchData *d) {
  if (d->rng_states) CheckHIPError(hipFree(d->rng_states));
  if (d->real_locations) CheckHIPError(hipFree(d->real_locations));
  if (d->imag_locations) CheckHIPError(hipFree(d->imag_locations));
  if (d->iterations_needed) CheckHIPError(hipFree(d->iterations_needed));
  memset(d, 0, sizeof(*d));
}

// This returns nonzero if the given point is in the main cardioid of the set
// and is therefore guaranteed to not escape.
__device__ int InMainCardioid(double real, double imag) {
  // This algorithm was taken from the Wikipedia Mandelbrot set page.
  double imag_squared = imag * imag;
  double q = (real - 0.25);
  q = q * q + imag_squared;
  return q * (q + (real - 0.25)) < (imag_squared * 0.25);
}

// This returns nonzero if the given point is in the order 2 bulb of the set
// and therefore guaranteed to not escape.
__device__ int InOrder2Bulb(double real, double imag) {
  double tmp = real + 1;
  tmp = tmp * tmp;
  return (tmp + (imag * imag)) < (1.0 / 16.0);
}

// Returns the number of iterations required for the given point to escape the
// mandelbrot set. Returns max_iterations if the point never escapes.
__device__ uint32_t GetMandelbrotIterations(double real, double imag,
    uint32_t max_iterations, double escape_radius) {
  if (InMainCardioid(real, imag)) return max_iterations;
  if (InOrder2Bulb(real, imag)) return max_iterations;
  uint32_t iteration = 0;
  double radius_squared = escape_radius * escape_radius;
  double start_real = real;
  double start_imag = imag;
  double tmp;
  for (iteration = 0; iteration < max_iterations; iteration++) {
    if (((real * real) + (imag * imag)) >= radius_squared) break;
    tmp = (real * real) - (imag * imag) + start_real;
    imag = 2 * real * imag + start_imag;
    real = tmp;
  }
  return iteration;
}

// Performs the random search for interesting Mandelbrot set locations. Fills
// in the location and iteration data if an interesting location is found.
__global__ void DoRandomSearch(RandomSearchData d) {
  int try_number;
  uint32_t iterations;
  double real, imag;
  uint32_t min_iterations = d.iterations.min_iterations;
  uint32_t max_iterations = d.iterations.max_iterations;
  int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  hiprandState_t *rng = d.rng_states + index;
  if (index > SEARCH_THREAD_COUNT) return;
  d.iterations_needed[index] = 0;
  d.real_locations[index] = -3;
  d.imag_locations[index] = -3;
  for (try_number = 0; try_number < RETRIES_PER_THREAD; try_number++) {
    // Get a random point within the 4x4 area at the center of the complex
    // plane.
    real = hiprand_uniform_double(rng) * 4.0 - 2.0;
    imag = hiprand_uniform_double(rng) * 4.0 - 2.0;
    iterations = GetMandelbrotIterations(real, imag, max_iterations,
      d.iterations.escape_radius);
    if (iterations < min_iterations) continue;
    if (iterations >= max_iterations) continue;
    // We found a point within the iteration bounds.
    d.iterations_needed[index] = iterations;
    d.real_locations[index] = real;
    d.imag_locations[index] = imag;
    return;
  }
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
  double radius_squared = m.iterations.escape_radius;
  radius_squared *= radius_squared;
  for (iteration = 0; iteration < m.iterations.max_iterations; iteration++) {
    if (((current_real * current_real) + (current_imag * current_imag)) >=
      radius_squared) {
      break;
    }
    tmp = (current_real * current_real) - (current_imag * current_imag) +
      start_real;
    current_imag = 2 * current_real * current_imag + start_imag;
    current_real = tmp;
  }
  m.data[index] = iteration;
}

// Returns the base-2 log of v, as an integer. This is equivalent to the
// position of the highest bit in v.
static uint32_t IntLog2(uint32_t v) {
  uint32_t to_return = 0;
  while (v != 0) {
    to_return++;
    v = v >> 1;
  }
  return to_return;
}

// Initializes the given image struct to render a Mandelbrot image centered on
// the given real, imaginary coordinate. Automatically calculates bounds based
// on the number of iterations needed.
static void InitializeImageBox(MandelbrotImage *image, double real,
    double imag, uint32_t max_iterations, int width) {
  double box_width = 0.0;
  // This only handles square images for now.
  InitializeImage(image, width, width);
  // We can leave min_iterations at 0, but set the other fields to ensure that
  // we'll get data at the spot we found randomly.
  image->iterations.max_iterations = max_iterations;
  image->iterations.min_iterations = 0;
  image->iterations.escape_radius = 2.1;

  // TODO: Actually calculate a dynamic box width.
  box_width = 1.0e-8;
  printf("Location at %f+%fi with %u iterations has box width %f\n",
    real, imag, (unsigned) max_iterations, box_width);
  image->dimensions.min_real = real - box_width / 2;
  image->dimensions.max_real = real + box_width / 2;
  image->dimensions.min_imag = imag - box_width / 2;
  image->dimensions.max_imag = imag + box_width / 2;
  UpdatePixelWidths(&(image->dimensions));
}

// Randomly searches for interesting Mandelbrot-set images. Requires a buffer
// of up to max_images MandelbrotImage structs to fill. Returns the number of
// images actually found. Requires an IterationControl object to constrain the
// search for interesting images.
static int GetRandomImages(MandelbrotImage *images, int max_images,
    IterationControl *iterations, int width) {
  RandomSearchData d;
  double *host_real = NULL;
  double *host_imag = NULL;
  uint32_t *host_iterations = NULL;
  int images_found = 0;
  int i = 0;
  size_t copy_size = 0;

  // First, perform the search for the random Mandelbrot images on the GPU.
  InitializeRandomSearchData(&d, iterations);
  int block_count = SEARCH_THREAD_COUNT / 256;
  if ((block_count % 256) != 0) block_count++;
  hipLaunchKernelGGL(DoRandomSearch, block_count, 256, 0, 0, d);
  CheckHIPError(hipDeviceSynchronize());

  // Next, copy the found image locations to the host.
  copy_size = SEARCH_THREAD_COUNT * sizeof(double);
  host_real = (double *) malloc(copy_size);
  if (!host_real) {
    printf("Failed allocating buffer for host real values.\n");
    return 0;
  }
  CheckHIPError(hipMemcpy(host_real, d.real_locations, copy_size,
    hipMemcpyDeviceToHost));
  host_imag = (double *) malloc(copy_size);
  if (!host_imag) {
    printf("Failed allocating buffer for host imaginary values.\n");
    free(host_real);
    return 0;
  }
  CheckHIPError(hipMemcpy(host_imag, d.imag_locations, copy_size,
    hipMemcpyDeviceToHost));
  copy_size = SEARCH_THREAD_COUNT * sizeof(uint32_t);
  host_iterations = (uint32_t *) malloc(copy_size);
  if (!host_iterations) {
    printf("Failed allocating buffer for host iteration counts.\n");
    free(host_real);
    free(host_imag);
    return 0;
  }
  CheckHIPError(hipMemcpy(host_iterations, d.iterations_needed, copy_size,
    hipMemcpyDeviceToHost));
  CleanupRandomSearchData(&d);

  // Next, iterate over the found locations and initialize the Mandelbrot-set
  // image structs with the correct locations. (Note that it's pointless to set
  // max_images to something higher than SEARCH_THREAD_COUNT).
  for (i = 0; i < SEARCH_THREAD_COUNT; i++) {
    if (host_iterations[i] == 0) continue;
    if (host_iterations[i] > iterations->max_iterations) continue;
    if ((host_real[i] < -2) || (host_real[i] > 2)) continue;
    if ((host_imag[i] < -2) || (host_imag[i] > 2)) continue;
    InitializeImageBox(images + images_found, host_real[i], host_imag[i],
      host_iterations[i], width);
    images_found++;
    if (images_found >= max_images) break;
  }
  free(host_real);
  free(host_imag);
  free(host_iterations);
  return images_found;
}

// Copies image data from the device buffer to a host buffer.
static void CopyResults(MandelbrotImage *m, uint32_t *host_data) {
  size_t byte_count = GetBufferSize(m);
  CheckHIPError(hipMemcpy(host_data, m->data, byte_count,
    hipMemcpyDeviceToHost));
}

static void GetHistogramColorData(MandelbrotImage *m, uint32_t *host_data,
    HistogramColorData *h) {
  uint32_t max_iterations = 0;
  uint64_t pixel_count = m->dimensions.w * m->dimensions.h;
  uint64_t *host_histogram = NULL;
  size_t histogram_size;
  uint64_t i;
  // Start by initializing the non-histogram fields of h.
  memset(h, 0, sizeof(*h));
  h->data = m->data;
  h->pixel_count = pixel_count;
  CheckHIPError(hipMalloc(&h->rgb_data, pixel_count * 3));

  // Find the maximum number of iterations, needed to allocate the histogram.
  for (i = 0; i < pixel_count; i++) {
    if (host_data[i] > max_iterations) max_iterations = host_data[i];
  }
  if (max_iterations == 0) {
    // We have zero iterations, and therefore need to histogram.
    return;
  }
  h->max_iterations = max_iterations;
  histogram_size = max_iterations * sizeof(uint64_t);
  host_histogram = (uint64_t *) malloc(histogram_size);
  if (!host_histogram) {
    printf("Failed allocating host-side histogram.\n");
    hipFree(h->rgb_data);
    memset(h, 0, sizeof(*h));
    return;
  }
  memset(host_histogram, 0, histogram_size);

  // Mext, calculate the histogram host-side and upload it to the device.
  for (i = 0; i < pixel_count; i++) {
    host_histogram[host_data[i]]++;
  }
  CheckHIPError(hipMalloc(&h->histogram, histogram_size));
  CheckHIPError(hipMemcpy(h->histogram, host_histogram, histogram_size,
    hipMemcpyHostToDevice));
  CheckHIPError(hipDeviceSynchronize());
  free(host_histogram);
  host_histogram = NULL;
}

// Frees the device memory associated with h, except for the raw image data,
// which should be cleaned up when the associated MandelbrotImage is cleaned
// up.
static void CleanupHistogramColorData(HistogramColorData *h) {
  if (h->histogram) CheckHIPError(hipFree(h->histogram));
  if (h->rgb_data) CheckHIPError(hipFree(h->rgb_data));
  memset(h, 0, sizeof(*h));
}

__global__ void ConvertIterationsToRGB(HistogramColorData h) {
  uint64_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  uint64_t rgb_start;
  uint32_t i, iterations;
  double v = 0.0;
  uint8_t gray_value;
  if (index > h.pixel_count) return;
  iterations = h.data[index];
  rgb_start = index * 3;
  for (i = 0; i < iterations; i++) {
    v += ((double) h.histogram[i]) / ((double) h.pixel_count);
  }
  v *= 255;
  if (v > 255) v = 255;
  if (v < 0) v = 0;
  gray_value = v;
  h.rgb_data[rgb_start] = v;
  h.rgb_data[rgb_start + 1] = v;
  h.rgb_data[rgb_start + 2] = v;
}

// Converts host data to color image data for writing to a PPM image.
static void GetRGBImage(MandelbrotImage *m, uint32_t *host_data,
    uint8_t *color_data) {
  uint64_t pixel_count;
  uint32_t block_count;
  HistogramColorData histogram_data;
  GetHistogramColorData(m, host_data, &histogram_data);
  pixel_count = histogram_data.pixel_count;
  block_count = pixel_count / 256;
  if ((block_count % 256) != 0) block_count++;
  hipLaunchKernelGGL(ConvertIterationsToRGB, block_count, 256, 0, 0,
    histogram_data);
  CheckHIPError(hipDeviceSynchronize());
  CheckHIPError(hipMemcpy(color_data, histogram_data.rgb_data, pixel_count * 3,
    hipMemcpyDeviceToHost));
  CleanupHistogramColorData(&histogram_data);
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

static void GenerateImages(int count, int width, const char *dir) {
  int number_found, i;
  char image_filename[1024];
  MandelbrotImage *images = NULL;
  IterationControl iterations;
  dim3 block_dim(16, 16);
  dim3 grid_dim((width / 16) + 1, (width / 16) + 1);
  memset(&iterations, 0, sizeof(iterations));
  iterations.max_iterations = 40000;
  iterations.min_iterations = 10000;
  iterations.escape_radius = 4.0;
  images = (MandelbrotImage *) malloc(count * sizeof(MandelbrotImage));
  if (!images) {
    printf("Failed allocating list of Mandelbrot images.\n");
    return;
  }
  number_found = GetRandomImages(images, count, &iterations, width);
  printf("Found %d images.\n", number_found);
  if (!number_found) {
    free(images);
    return;
  }
  for (i = 0; i < number_found; i++) {
    memset(image_filename, 0, sizeof(image_filename));
    snprintf(image_filename, sizeof(image_filename), "%s/%d.ppm", dir, i);
    printf("Rendering image %d of %d...\n", i, number_found);
    hipLaunchKernelGGL(DrawMandelbrot, grid_dim, block_dim, 0, 0, images[i]);
    CheckHIPError(hipDeviceSynchronize());
    if (!SaveImage(images + i, image_filename)) {
      printf("Failed saving image %s\n", image_filename);
    } else {
      printf("Image saved as %s\n", image_filename);
    }
    CleanupImage(images + i);
  }
  free(images);
  images = NULL;
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
  int count, resolution;
  MandelbrotImage m;
  if (argc != 4) {
    printf("Usage: %s <max # of images> <image width> <output directory>\n",
      argv[0]);
    return 1;
  }
  count = atoi(argv[1]);
  if (count <= 0) {
    printf("Bad image count: %s\n", argv[1]);
    return 1;
  }
  resolution = atoi(argv[2]);
  if (resolution <= 0) {
    printf("Bad image width: %s\n", argv[2]);
    return 1;
  }
  SetupDevice();
  GenerateImages(count, resolution, argv[3]);
  return 0;
}
