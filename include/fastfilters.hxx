#ifndef FASTFILTERS_HXX
#define FASTFILTERS_HXX

#include <array>
#include <vector>

#if defined(_MSC_VER)
#if defined(FASTFILTERS_SHARED_LIBRARY)
#define FASTFILTERS_API_EXPORT __declspec(dllexport)
#elif defined(FASTFILTERS_STATIC_LIBRARY)
#define FASTFILTERS_API_EXPORT
#else
#define FASTFILTERS_API_EXPORT __declspec(dllimport)
#endif
#else
#define FASTFILTERS_API_EXPORT
#endif

namespace fastfilters
{

namespace detail
{
FASTFILTERS_API_EXPORT bool cpu_has_avx2();
}

namespace fir
{

struct FASTFILTERS_API_EXPORT Kernel
{
    const bool is_symmetric;
    const std::vector<float> coefs;
    std::vector<float> coefs2;

    inline Kernel(bool is_symmetric, const std::vector<float> &coefs) : is_symmetric(is_symmetric), coefs(coefs)
    {
        coefs2 = std::vector<float>(len());

        for (unsigned int idx = 0; idx < len(); ++idx) {
            float v;

            if (idx == half_len())
                v = coefs[0];
            else if (idx < half_len()) {
                if (is_symmetric)
                    v = coefs[half_len() - idx];
                else
                    v = -coefs[half_len() - idx];
            } else
                v = coefs[idx - half_len()];
            coefs2[idx] = v;
        }
    }

    inline float operator[](std::size_t idx) const
    {
        return coefs2[idx];
    };

    inline std::size_t len() const
    {
        return 2 * coefs.size() - 1;
    }
    inline std::size_t half_len() const
    {
        return coefs.size() - 1;
    }
};

FASTFILTERS_API_EXPORT void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned n_times, const unsigned int dim_stride,
                                                          float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned int pixel_stride, const unsigned n_times,
                                                          float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned n_times, const unsigned int dim_stride,
                                                            float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned int pixel_stride, const unsigned n_times,
                                                            const unsigned dim_stride, float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_inner_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned n_times, const unsigned int dim_stride,
                                                      float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned int pixel_stride, const unsigned n_times,
                                                      const unsigned int dim_stride, float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir(const float *input, const unsigned int pixel_stride,
                                         const unsigned int pixel_n, const unsigned int dim_stride,
                                         const unsigned int dim, float *output, Kernel &kernel);
}

namespace iir
{

struct FASTFILTERS_API_EXPORT Coefficients
{
    std::array<float, 4> n_causal;
    std::array<float, 4> n_anticausal;
    std::array<float, 4> d;
    double sigma;
    unsigned order;
    unsigned n_border;

    Coefficients(const double sigma, const unsigned order);
};

FASTFILTERS_API_EXPORT void convolve_iir_inner_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned n_times, float *output,
                                                            const Coefficients &coefs);

FASTFILTERS_API_EXPORT void convolve_iir_outer_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned n_times, float *output,
                                                            const Coefficients &coefs, const unsigned stride);

FASTFILTERS_API_EXPORT void convolve_iir_inner_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned n_times, float *output,
                                                          const Coefficients &coefs);

FASTFILTERS_API_EXPORT void convolve_iir_outer_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned n_times, float *output,
                                                          const Coefficients &coefs);

FASTFILTERS_API_EXPORT void convolve_iir_inner_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned n_times, float *output, const Coefficients &coefs);

FASTFILTERS_API_EXPORT void convolve_iir_outer_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned n_times, float *output, const Coefficients &coefs);

} // namespace iir

} // namespace fastfilters

#endif