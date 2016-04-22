// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#ifndef FASTFILTERS_H
#define FASTFILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
#ifdef FASTFILTERS_SHARED_LIBRARY
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllexport))
#else
#define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
#endif
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllimport))
#else
#define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
#endif
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif

typedef struct _fastfilters_kernel_fir_t *fastfilters_kernel_fir_t;

typedef enum { FASTFILTERS_CPU_AVX, FASTFILTERS_CPU_FMA, FASTFILTERS_CPU_AVX2 } fastfilters_cpu_feature_t;

typedef struct _fastfilters_array2d_t {
    float *ptr;
    size_t n_x;
    size_t n_y;
    size_t stride_x;
    size_t stride_y;
    size_t n_channels;
} fastfilters_array2d_t;

typedef struct _fastfilters_array3d_t {
    float *ptr;
    size_t n_x;
    size_t n_y;
    size_t n_z;
    size_t stride_x;
    size_t stride_y;
    size_t stride_z;
    size_t n_channels;
} fastfilters_array3d_t;

typedef void *(*fastfilters_alloc_fn_t)(size_t size);
typedef void (*fastfilters_free_fn_t)(void *);

void DLL_PUBLIC fastfilters_init(fastfilters_alloc_fn_t alloc_fn, fastfilters_free_fn_t free_fn);

bool DLL_PUBLIC fastfilters_cpu_check(fastfilters_cpu_feature_t feature);
bool DLL_PUBLIC fastfilters_cpu_enable(fastfilters_cpu_feature_t feature, bool enable);

fastfilters_kernel_fir_t DLL_PUBLIC fastfilters_kernel_fir_gaussian(unsigned int order, double sigma);
unsigned int DLL_PUBLIC fastfilters_kernel_fir_get_length(fastfilters_kernel_fir_t kernel);
void DLL_PUBLIC fastfilters_kernel_fir_free(fastfilters_kernel_fir_t kernel);

bool DLL_PUBLIC fastfilters_fir_convolve2d(const fastfilters_array2d_t *inarray, const fastfilters_kernel_fir_t kernelx,
                                           const fastfilters_kernel_fir_t kernely,
                                           const fastfilters_array2d_t *outarray, size_t x0, size_t y0, size_t x1,
                                           size_t y1);
bool DLL_PUBLIC fastfilters_fir_convolve3d(const fastfilters_array3d_t *inarray, const fastfilters_kernel_fir_t kernelx,
                                           const fastfilters_kernel_fir_t kernely,
                                           const fastfilters_kernel_fir_t kernelz,
                                           const fastfilters_array2d_t *outarray, size_t x0, size_t y0, size_t z0,
                                           size_t x1, size_t y1, size_t z1);

void DLL_PUBLIC fastfilters_linalg_ev2d(const float *xx, const float *xy, const float *yy, float *ev_small,
                                        float *ev_big, const size_t len);

void DLL_PUBLIC fastfilters_combine_add2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                          fastfilters_array2d_t *out);
void DLL_PUBLIC fastfilters_combine_addsqrt2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                              fastfilters_array2d_t *out);
void DLL_PUBLIC fastfilters_combine_mul2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                          fastfilters_array2d_t *out);

DLL_PUBLIC fastfilters_array2d_t *fastfilters_array2d_alloc(size_t n_x, size_t n_y, size_t channels);
DLL_PUBLIC void fastfilters_array2d_free(fastfilters_array2d_t *v);

bool DLL_PUBLIC fastfilters_fir_gaussian2d(const fastfilters_array2d_t *inarray, unsigned order, double sigma,
                                           fastfilters_array2d_t *outarray);

bool DLL_PUBLIC fastfilters_fir_hog2d(const fastfilters_array2d_t *inarray, double sigma, fastfilters_array2d_t *out_xx,
                                      fastfilters_array2d_t *out_xy, fastfilters_array2d_t *out_yy);

bool DLL_PUBLIC fastfilters_fir_gradmag2d(const fastfilters_array2d_t *inarray, double sigma,
                                          fastfilters_array2d_t *outarray);

bool DLL_PUBLIC fastfilters_fir_laplacian2d(const fastfilters_array2d_t *inarray, double sigma,
                                            fastfilters_array2d_t *outarray);
bool DLL_PUBLIC fastfilters_fir_structure_tensor(const fastfilters_array2d_t *inarray, double sigma_outer,
                                                 double sigma_inner, fastfilters_array2d_t *out_xx,
                                                 fastfilters_array2d_t *out_xy, fastfilters_array2d_t *out_yy);
#ifdef __cplusplus
}
#endif

#endif