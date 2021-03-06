#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include "hip_rng.h"

// The device number on which to perform the computation.
#define DEVICE_ID (0)

// The number of GPU threads to use when searching for interesting locations.
#define SEARCH_THREAD_COUNT (1024 * 32)

// The number of times each thread retries searching for a good location.
#define RETRIES_PER_THREAD (256)

// The mandelbrot escape radius. 2 is sufficient, but higher numbers result in
// smoother coloring.
#define ESCAPE_RADIUS (64.0)

// This determines the amount of supersamples to use when drawing the set. For
// example, if this is 3, then each pixel in the output image will be
// determined by the average iteration of a 3x3 box centered on the original
// complex point.
#define SUPERSAMPLING_AMOUNT (2)

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
} IterationControl;

// This holds all data and settings for rendering a Mandelbrot set.
typedef struct {
  FractalDimensions dimensions;
  IterationControl iterations;
  // This is the data *on the device*.
  double *data;
} MandelbrotImage;

// This contains parameters and outputs used when searching for interesting
// locations.
typedef struct {
  // A list of RNG states for the search. There must be one such state per
  // search thread.
  RNGState *rng_states;
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
  double *histogram;
  // The number of entries in the raw data buffer
  uint64_t pixel_count;
  // The original raw image data
  double *data;
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

static double GetTimeSeconds(void) {
  double to_return = 0;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  to_return = t.tv_sec;
  to_return += t.tv_nsec * 1e-9;
  return to_return;
}

// Fills the buffer with random lowercase alphanumeric characters. Places a
// null character at the end of the buffer.
static void GetRandomString(char *buffer, int buffer_size) {
  int i;
  for (i = 0; i < (buffer_size - 1); i++) {
    buffer[i] = 'a' + (rand() % 26);
  }
  buffer[buffer_size - 1] = 0;
}

static void CleanupImage(MandelbrotImage *m) {
  if (m->data) hipFree(m->data);
  memset(m, 0, sizeof(*m));
}

// This returns the number of bytes needed to hold the 32-bit array of
// iterations for each output pixel.
static size_t GetBufferSize(MandelbrotImage *m) {
  return m->dimensions.w * m->dimensions.h * sizeof(double);
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
  UpdatePixelWidths(&(m->dimensions));
  AllocateDeviceMemory(m);
}

__global__ void InitializeRNG(uint64_t seed, RNGState *states) {
  int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (index > SEARCH_THREAD_COUNT) return;
  InitRNGState(seed, index, states + index);
}

// Initializes the GPU memory and RNG states for the random search.
static void InitializeRandomSearchData(RandomSearchData *d,
    IterationControl *c) {
  int count = SEARCH_THREAD_COUNT;
  int block_count = count / 256;
  if ((count % 256) != 0) block_count++;
  CheckHIPError(hipMalloc(&(d->rng_states), count * sizeof(RNGState)));
  CheckHIPError(hipMalloc(&(d->real_locations), count * sizeof(double)));
  CheckHIPError(hipMemset(d->real_locations, 0, count * sizeof(double)));
  CheckHIPError(hipMalloc(&(d->imag_locations), count * sizeof(double)));
  CheckHIPError(hipMemset(d->imag_locations, 0, count * sizeof(double)));
  CheckHIPError(hipMalloc(&(d->iterations_needed), count * sizeof(uint32_t)));
  CheckHIPError(hipMemset(d->iterations_needed, 0, count * sizeof(uint32_t)));
  CheckHIPError(hipDeviceSynchronize());
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
__device__ double GetMandelbrotIterations(double real, double imag,
    uint32_t max_iterations) {
  if (InMainCardioid(real, imag)) return max_iterations;
  if (InOrder2Bulb(real, imag)) return max_iterations;
  uint32_t iteration = 0;
  double radius_squared = ESCAPE_RADIUS * ESCAPE_RADIUS;
  double start_real, start_imag, tmp, smooth;
  start_real = real;
  start_imag = imag;
  for (iteration = 0; iteration < max_iterations; iteration++) {
    if (((real * real) + (imag * imag)) >= radius_squared) break;
    tmp = (real * real) - (imag * imag) + start_real;
    imag = 2 * real * imag + start_imag;
    real = tmp;
  }
  if (iteration >= max_iterations) return max_iterations;
  smooth = iteration;
  smooth += 1 - log2(log(sqrt((real * real) + (imag * imag))));
  return smooth;
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
  RNGState *rng = d.rng_states + index;
  if (index > SEARCH_THREAD_COUNT) return;
  d.iterations_needed[index] = 0;
  d.real_locations[index] = -3;
  d.imag_locations[index] = -3;
  for (try_number = 0; try_number < RETRIES_PER_THREAD; try_number++) {
    // Get a random point within the 4x4 area at the center of the complex
    // plane.
    real = UniformDouble(rng) * 4.0 - 2.0;
    imag = UniformDouble(rng) * 4.0 - 2.0;
    iterations = GetMandelbrotIterations(real, imag, max_iterations);
    if (iterations < min_iterations) continue;
    if (iterations >= max_iterations) continue;
    // We found a point within the iteration bounds.
    d.iterations_needed[index] = iterations;
    d.real_locations[index] = real;
    d.imag_locations[index] = imag;
    return;
  }
}

__device__ double Average(double *array, int count) {
  double total = 0.0;
  int i;
  for (i = 0; i < count; i++) {
    total += array[i];
  }
  return total / (double) count;
}

// This uses a 2-D grid of threads. The x and y dimensions must cover the
// entire width of the output image.
__global__ void DrawMandelbrot(MandelbrotImage m) {
  int x, y, col, row, index;
  double start_real, real, imag, actual_dx, actual_dy;
  double supersamples[SUPERSAMPLING_AMOUNT * SUPERSAMPLING_AMOUNT];
  int sample_count = SUPERSAMPLING_AMOUNT * SUPERSAMPLING_AMOUNT;
  col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  if (col >= m.dimensions.w) return;
  if (row >= m.dimensions.h) return;
  actual_dx = m.dimensions.delta_real / SUPERSAMPLING_AMOUNT;
  actual_dy = m.dimensions.delta_imag / SUPERSAMPLING_AMOUNT;
  real = m.dimensions.delta_real * col + m.dimensions.min_real;
  imag = m.dimensions.delta_imag * row + m.dimensions.min_imag;
  // Offset the start real and imaginary coordinates to center the samples on
  // the original point.
  real -= actual_dx * ((double) SUPERSAMPLING_AMOUNT / 2.0);
  imag -= actual_dy * ((double) SUPERSAMPLING_AMOUNT / 2.0);
  start_real = real;
  for (y = 0; y < SUPERSAMPLING_AMOUNT; y++) {
    for (x = 0; x < SUPERSAMPLING_AMOUNT; x++) {
      index = y * SUPERSAMPLING_AMOUNT + x;
      supersamples[index] = GetMandelbrotIterations(real, imag,
        m.iterations.max_iterations);
      real += actual_dx;
    }
    imag += actual_dy;
    real = start_real;
  }
  index = row * m.dimensions.w + col;
  m.data[index] = Average(supersamples, sample_count);
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

  // TODO: Actually calculate a dynamic box width.
  box_width = 1.0e-12;
  printf("Location at %f+%fi with %u iterations\n", real, imag,
    (unsigned) max_iterations);
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
  double start_time = GetTimeSeconds();

  // First, perform the search for the random Mandelbrot images on the GPU.
  InitializeRandomSearchData(&d, iterations);
  int block_count = SEARCH_THREAD_COUNT / 256;
  if ((block_count % 256) != 0) block_count++;
  hipLaunchKernelGGL(DoRandomSearch, block_count, 256, 0, 0, d);
  CheckHIPError(hipDeviceSynchronize());


  // Next, copy the found image locations to the host.
  copy_size = SEARCH_THREAD_COUNT * sizeof(double);
  CheckHIPError(hipHostMalloc(&host_real, copy_size));
  CheckHIPError(hipMemcpy(host_real, d.real_locations, copy_size,
    hipMemcpyDeviceToHost));
  CheckHIPError(hipHostMalloc(&host_imag, copy_size));
  CheckHIPError(hipMemcpy(host_imag, d.imag_locations, copy_size,
    hipMemcpyDeviceToHost));
  copy_size = SEARCH_THREAD_COUNT * sizeof(uint32_t);
  CheckHIPError(hipHostMalloc(&host_iterations, copy_size));
  CheckHIPError(hipMemcpy(host_iterations, d.iterations_needed, copy_size,
    hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
  CleanupRandomSearchData(&d);

  printf("The random search took %f seconds.\n", GetTimeSeconds() - start_time);

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
  CheckHIPError(hipHostFree(host_real));
  CheckHIPError(hipHostFree(host_imag));
  CheckHIPError(hipHostFree(host_iterations));
  return images_found;
}

// Copies image data from the device buffer to a host buffer.
static void CopyResults(MandelbrotImage *m, double *host_data) {
  size_t byte_count = GetBufferSize(m);
  CheckHIPError(hipMemcpy(host_data, m->data, byte_count,
    hipMemcpyDeviceToHost));
}

static void GetHistogramColorData(MandelbrotImage *m, double *host_data,
    HistogramColorData *h) {
  uint64_t i, max_iterations, tmp;
  double fractional_iterations;
  uint64_t pixel_count = m->dimensions.w * m->dimensions.h;
  size_t histogram_size;
  double *host_histogram = NULL;
  // Start by initializing the non-histogram fields of h.
  memset(h, 0, sizeof(*h));
  h->data = m->data;
  h->pixel_count = pixel_count;
  CheckHIPError(hipMalloc(&h->rgb_data, pixel_count * 3));

  // Find the maximum number of iterations, needed to allocate the histogram.
  max_iterations = 0;
  for (i = 0; i < pixel_count; i++) {
    tmp = llround(ceil(host_data[i]));
    if (tmp > max_iterations) max_iterations = tmp;
  }
  if (max_iterations == 0) {
    // We have zero iterations, and therefore need no histogram.
    return;
  }
  h->max_iterations = max_iterations;
  histogram_size = (max_iterations + 1) * sizeof(double);
  CheckHIPError(hipHostMalloc(&host_histogram, histogram_size));
  memset(host_histogram, 0, histogram_size);

  // Next, calculate the histogram host-side.
  for (i = 0; i < pixel_count; i++) {
    tmp = llround(floor(host_data[i]));
    fractional_iterations = host_data[i] - ((double) tmp);
    host_histogram[tmp] += 1;
    host_histogram[tmp + 1] += fractional_iterations;
  }

  // Finally, upload the histogram to the device.
  CheckHIPError(hipMalloc(&h->histogram, histogram_size));
  CheckHIPError(hipMemcpy(h->histogram, host_histogram, histogram_size,
    hipMemcpyHostToDevice));
  CheckHIPError(hipDeviceSynchronize());
  CheckHIPError(hipHostFree(host_histogram));
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
  uint64_t i;
  double iterations;
  double v = 0.0;
  double total;
  uint32_t gray_value;
  if (index > h.pixel_count) return;
  total = (double) h.pixel_count;
  iterations = h.data[index];
  rgb_start = index * 3;
  for (i = 0; i <= iterations; i++) {
    v += h.histogram[i] / total;
  }
  // Add a value for the fractional iterations
  v += (h.histogram[i + 1] * (iterations - i)) / total;
  v *= 255;
  gray_value = lround(v);
  if (gray_value > 255) gray_value = 255;
  if (gray_value < 0) gray_value = 0;
  h.rgb_data[rgb_start] = gray_value;
  h.rgb_data[rgb_start + 1] = gray_value;
  h.rgb_data[rgb_start + 2] = gray_value;
}

// Converts host data to color image data for writing to a PPM image.
static void GetRGBImage(MandelbrotImage *m, double *host_data,
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
  CheckHIPError(hipDeviceSynchronize());
  CleanupHistogramColorData(&histogram_data);
}

// Saves a ppm-format file to the given filename. Returns 0 on error and
// nonzero otherwise.
static int SaveImage(MandelbrotImage *m, const char *filename) {
  FractalDimensions dims = m->dimensions;
  FILE *f = NULL;
  uint8_t *color_data = NULL;
  size_t color_data_size = 0;
  double *host_data = NULL;

  // First, copy the raw output buffer to the host.
  CheckHIPError(hipHostMalloc(&host_data, GetBufferSize(m)));
  CopyResults(m, host_data);

  // Next, convert the raw buffer to RGB pixel data. We need 3 bytes per pixel
  // here.
  color_data_size = dims.w * dims.h * 3;
  CheckHIPError(hipHostMalloc(&color_data, color_data_size));
  GetRGBImage(m, host_data, color_data);
  CheckHIPError(hipHostFree(host_data));
  host_data = NULL;

  // Next, create the output file and write the data to it.
  f = fopen(filename, "wb");
  if (!f) {
    printf("Failed opening output file: %s\n", strerror(errno));
    hipHostFree(color_data);
    return 0;
  }
  // Include the parameters for the set as a "comment" in the image.
  if (fprintf(f, "P6\n# Bounds: (%f, %fi, %f, %fi). Max iters: %u\n",
    dims.min_real, dims.min_imag, dims.max_real, dims.max_imag,
    (unsigned) m->iterations.max_iterations) <= 0) {
    printf("Failed writing Mandelbrot metadata: %s\n", strerror(errno));
    fclose(f);
    hipHostFree(color_data);
    return 0;
  }
  if (fprintf(f, "%d %d\n255\n", dims.w, dims.h) <= 0) {
    printf("Failed writing ppm header: %s\n", strerror(errno));
    fclose(f);
    hipHostFree(color_data);
    return 0;
  }

  if (!fwrite(color_data, color_data_size, 1, f)) {
    printf("Failed writing pixel data: %s\n", strerror(errno));
    fclose(f);
    free(color_data);
    hipHostFree(color_data);
    return 0;
  }
  fclose(f);
  CheckHIPError(hipHostFree(color_data));
  return 1;
}

// Appends a line about the given Mandelbrot image to the given log file, so
// that the parameters for the given image name can be found if desired.
// Returns 0 on error.
static int AppendMetadataToLog(MandelbrotImage *m, char *name, FILE *f) {
  FractalDimensions dims = m->dimensions;
  if (fprintf(f, "%s: Bounds = %.14f+%.14fi to %.14f+%.14fi. "
    "Max iterations: %u.\n", name, dims.min_real, dims.min_imag, dims.max_real,
    dims.max_imag, m->iterations.max_iterations) <= 0) {
    printf("Failed writing image info to log file: %s\n", strerror(errno));
    return 0;
  }
  return 1;
}

static void GenerateImages(int count, int width, const char *dir) {
  int number_found, i;
  char image_filename[1024];
  char random_name[16];
  FILE *info_log = NULL;
  MandelbrotImage *images = NULL;
  IterationControl iterations;
  dim3 block_dim(16, 16);
  dim3 grid_dim((width / 16) + 1, (width / 16) + 1);
  memset(&iterations, 0, sizeof(iterations));
  iterations.max_iterations = 40000;
  iterations.min_iterations = 10000;
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
  snprintf(image_filename, sizeof(image_filename), "%s/image_parameters.txt",
    dir);
  info_log = fopen(image_filename, "ab");
  if (!info_log) {
    printf("Failed opening info log %s: %s. Continuing without logging.\n",
      image_filename, strerror(errno));
  }
  for (i = 0; i < number_found; i++) {
    memset(image_filename, 0, sizeof(image_filename));
    GetRandomString(random_name, sizeof(random_name));
    snprintf(image_filename, sizeof(image_filename), "%s/%d_%s.ppm", dir,
      i + 1, random_name);
    printf("Rendering image %d of %d...\n", i + 1, number_found);
    hipLaunchKernelGGL(DrawMandelbrot, grid_dim, block_dim, 0, 0, images[i]);
    CheckHIPError(hipDeviceSynchronize());
    if (!SaveImage(images + i, image_filename)) {
      printf("Failed saving image %s\n", image_filename);
    } else {
      printf("Image saved as %s\n", image_filename);
    }
    if (info_log) AppendMetadataToLog(images + i, image_filename, info_log);
    CleanupImage(images + i);
  }
  if (info_log) fclose(info_log);
  info_log = NULL;
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
  srand(GetRNGSeed());
  SetupDevice();
  GenerateImages(count, resolution, argv[3]);
  return 0;
}
