#ifndef FASTFILTERS_HXX
#define FASTFILTERS_HXX

#include <array>

namespace fastfilters
{

namespace fir
{
void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const float *kernel, const unsigned int kernel_len);

void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const float *kernel, const unsigned int kernel_len);

void convolve_fir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const float *kernel, const unsigned int kernel_len);
}

namespace iir
{

struct Coefficients
{
    std::array<float, 4> n_causal;
    std::array<float, 4> n_anticausal;
    std::array<float, 4> d;
    double sigma;
    unsigned order;
    unsigned n_border;

    Coefficients(const double sigma, const unsigned order);
};

void convolve_iir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs);

void convolve_iir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs, const unsigned stride);

void convolve_iir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs);

void convolve_iir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs);
}

namespace detail
{
bool cpu_has_avx2();
}

} // namespace fastfilters

#endif