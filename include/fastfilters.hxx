#ifndef FASTFILTERS_HXX
#define FASTFILTERS_HXX

#include <array>
#include <vector>

namespace fastfilters
{

namespace detail
{
bool cpu_has_avx2();
}

namespace fir
{

struct Kernel
{
    const bool is_symmetric;
    const std::vector<float> coefs;

    Kernel(bool is_symmetric, const std::vector<float> &coefs) : is_symmetric(is_symmetric), coefs(coefs)
    {
    }

    float operator[](std::size_t idx) const
    {
        if (idx == half_len())
            return coefs[0];

        if (idx < half_len()) {
            if (is_symmetric)
                return coefs[half_len() - idx];
            else
                return -coefs[half_len() - idx];
        }

        return coefs[idx - half_len()];
    };

    std::size_t len() const
    {
        return 2 * coefs.size() - 1;
    }
    std::size_t half_len() const
    {
        return coefs.size() - 1;
    }
};

void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   const unsigned int dim_stride, float *output, Kernel &kernel);

void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                                   const unsigned n_times, float *output, Kernel &kernel);

void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     const unsigned int dim_stride, float *output, Kernel &kernel);

void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                                     const unsigned n_times, const unsigned dim_stride, float *output, Kernel &kernel);

void convolve_fir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times,
                               const unsigned int dim_stride, float *output, Kernel &kernel);

void convolve_fir_outer_single(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                               const unsigned n_times, const unsigned int dim_stride, float *output, Kernel &kernel);

void convolve_fir(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                  const unsigned int dim_stride, const unsigned int dim, float *output, Kernel &kernel);
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

void convolve_iir_inner_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     float *output, const Coefficients &coefs);

void convolve_iir_outer_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     float *output, const Coefficients &coefs, const unsigned stride);

void convolve_iir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs);

void convolve_iir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs);

void convolve_iir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs);

void convolve_iir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs);

} // namespace iir

} // namespace fastfilters

#endif