#ifndef FASTFILTERS_HXX
#define FASTFILTERS_HXX

#include <array>

namespace fastfilters
{

template <unsigned N> class FastFilterArrayView
{
  public:
    float *baseptr;
    const unsigned int n_pixels[N];
    const unsigned int n_channels;

    FastFilterArrayView(float *baseptr, const unsigned int n_pixels[N], const unsigned int n_channels)
        : baseptr(baseptr), n_pixels(n_pixels), n_channels(n_channels)
    {
    }
};

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

    Coefficients(const double sigma, const unsigned order);
};

void convolve_iir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs, const unsigned n_border);

void convolve_iir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs, const unsigned n_border, const unsigned stride);

void convolve_iir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs, const unsigned n_border);

void convolve_iir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs, const unsigned n_border);
}

namespace detail
{
bool cpu_has_avx2();

template <unsigned N>
void gaussian_fir_inner(FastFilterArrayView<N> &input, FastFilterArrayView<N> &output, const unsigned order,
                        const double sigma)
{
    (void)input;
    (void)output;
    (void)sigma;
    (void)order;
}

template <unsigned N>
void gaussian_fir_outer(FastFilterArrayView<N> &input, const unsigned n_dim, FastFilterArrayView<N> &output,
                        const unsigned order, const double sigma)
{
    (void)input;
    (void)output;
    (void)sigma;
    (void)n_dim;
    (void)order;
}

template <unsigned N>
void gaussian_iir_inner(FastFilterArrayView<N> &input, FastFilterArrayView<N> &output, const unsigned order,
                        const double sigma)
{
    (void)input;
    (void)output;
    (void)sigma;
    (void)order;
}

template <unsigned N>
void gaussian_iir_outer(FastFilterArrayView<N> &input, const unsigned n_dim, FastFilterArrayView<N> &output,
                        const unsigned order, const double sigma)
{
    (void)input;
    (void)output;
    (void)sigma;
    (void)n_dim;
    (void)order;
}

template <unsigned N>
void gaussian_inner(FastFilterArrayView<N> &input, FastFilterArrayView<N> &output, const unsigned order,
                    const double sigma)
{
    if (sigma < 3)
        gaussian_fir_inner(input, output, order, sigma);
    else
        gaussian_iir_inner(input, output, order, sigma);
}

template <unsigned N>
void gaussian_outer(FastFilterArrayView<N> &input, const unsigned n_dim, FastFilterArrayView<N> &output,
                    const unsigned order, const double sigma)
{
    if (sigma < 3)
        gaussian_fir_outer(input, n_dim, output, order, sigma);
    else
        gaussian_iir_outer(input, n_dim, output, order, sigma);
}
} // namespace detail

template <unsigned N>
void gaussian(FastFilterArrayView<N> &input, FastFilterArrayView<N> &output, const unsigned order, const double sigma)
{
    detail::gaussian_inner(input, output, order, sigma);

    for (unsigned int i = 1; i < N; ++i)
        detail::gaussian_outer(input, i, output, order, sigma);
}

} // namespace fastfilters

#endif