Randobrot: A random Mandelbrot-set renderer in HIP
==================================================

This project was just intended as a way for me to learn how to use AMD's HIP.

Prerequisites
-------------

You need to have AMD's `hipcc` installed, as well as hipRAND. `hipcc` must be
on your `PATH`.

Usage
---------

Assuming the prerequisites are set up, run `make`.

To generate 12 images of 1000x1000 Mandelbrot sets, run:

```bash
./randobrot 12 1000 ./output_directory/
```

This will generate 12 .ppm images with random names in the output directory.
Each image will include a comment line before the image data, containing the
boundary in the complex plane and number of iterations used to render the
image.

