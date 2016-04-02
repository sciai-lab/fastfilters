#include "fastfilters.hxx"

namespace fastfilters
{

namespace iir
{
void convolve_iir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs)
{
    if (detail::cpu_has_avx2())
        convolve_iir_inner_single_avx(input, n_pixels, n_times, output, coefs);
    else
        convolve_iir_inner_single_noavx(input, n_pixels, n_times, output, coefs);
}

void convolve_iir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs)
{
    if (detail::cpu_has_avx2())
        convolve_iir_outer_single_avx(input, n_pixels, n_times, output, coefs);
    else
        convolve_iir_outer_single_noavx(input, n_pixels, n_times, output, coefs, n_times);
}

} // namepsace iir

} // namespace fastfilters