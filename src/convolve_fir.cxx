#include "fastfilters.hxx"

namespace fastfilters
{
namespace fir
{

void convolve_fir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times,
                               const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (detail::cpu_has_avx2())
        convolve_fir_inner_single_avx(input, n_pixels, n_times, dim_stride, output, kernel);
    else
        convolve_fir_inner_single_noavx(input, n_pixels, n_times, dim_stride, output, kernel);
}

void convolve_fir_outer_single(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                               const unsigned n_times, const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (detail::cpu_has_avx2() && dim_stride == 1)
        convolve_fir_outer_single_avx(input, n_pixels, pixel_stride, n_times, output, kernel);
    else
        convolve_fir_outer_single_noavx(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
}

void convolve_fir(const float *input, const unsigned int pixel_n, const unsigned int pixel_stride,
                  const unsigned int dim_n, const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (pixel_stride == 1)
        convolve_fir_inner_single(input, pixel_n, dim_n, dim_stride, output, kernel);
    else
        convolve_fir_outer_single(input, pixel_n, pixel_stride, dim_n, dim_stride, output, kernel);
}

} // namespace fir

} // namespace fastfilters