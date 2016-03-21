#include "fastfilters.hxx"

#include <vector>

// the AVX versions need these scalar functions for up to the last 7 values.
// this hack allows these functions to be compiled with much
// more agressive optimizations (such as enabling the
// fused multiply-add instructions)
#ifndef CONVOLVE_IIR_FUNCTION
#define CONVOLVE_IIR_FUNCTION(x) void x
#endif

namespace fastfilters
{

namespace iir
{

CONVOLVE_IIR_FUNCTION(convolve_iir_inner_single)(const float *input, const unsigned int n_pixels,
                                                 const unsigned n_times, float *output,
                                                 const std::array<float, 4> &n_causal,
                                                 const std::array<float, 4> &n_anticausal,
                                                 const std::array<float, 4> &d, const unsigned n_border)
{
    std::vector<float> tmpbfr(n_pixels);

    for (unsigned int dim = 0; dim < n_times; dim++) {
        const float *cur_line = input + dim * n_pixels;
        float *cur_output = output + dim * n_pixels;
        float xtmp[4];
        float ytmp[4];

        for (unsigned int i = 0; i < 4; ++i)
            xtmp[i] = ytmp[i] = 0.0;

        // left border
        for (unsigned int i = 0; i < n_border; ++i) {
            float sum = 0.0;

            xtmp[0] = cur_line[n_border - i];
            for (unsigned int j = 0; j < 4; ++j)
                sum += n_causal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            ytmp[0] = sum;
        }

        // causal pass
        for (unsigned int i = 0; i < n_pixels; ++i) {
            float sum = 0.0;

            xtmp[0] = cur_line[i];
            for (unsigned int j = 0; j < 4; ++j)
                sum += n_causal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            tmpbfr[i] = sum;
            ytmp[0] = sum;
        }

        // reset variables for anti-causal pass
        for (unsigned int i = 0; i < 4; ++i)
            xtmp[i] = ytmp[i] = 0.0;

        // right border
        for (int i = n_border; i > 0; --i) {
            float sum = 0.0;

            for (unsigned int j = 0; j < 4; ++j)
                sum += n_anticausal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            xtmp[0] = cur_line[n_pixels - i];
            ytmp[0] = sum;
        }

        // anti-causal pass
        for (int i = n_pixels - 1; i >= 0; --i) {
            float sum = 0.0;

            for (unsigned int j = 0; j < 4; ++j)
                sum += n_anticausal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            xtmp[0] = cur_line[i];
            ytmp[0] = sum;
            cur_output[i] = tmpbfr[i] + sum;
        }
    }
}

CONVOLVE_IIR_FUNCTION(convolve_iir_outer_single)(const float *input, const unsigned int n_pixels,
                                                 const unsigned n_times, float *output,
                                                 const std::array<float, 4> &n_causal,
                                                 const std::array<float, 4> &n_anticausal,
                                                 const std::array<float, 4> &d, const unsigned n_border,
                                                 unsigned int stride)
{
    std::vector<float> tmpbfr(n_pixels);

    for (unsigned int dim = 0; dim < n_times; dim++) {
        const float *cur_line = input + dim;
        float *cur_output = output + dim;
        float xtmp[4];
        float ytmp[4];

        for (unsigned int i = 0; i < 4; ++i)
            xtmp[i] = ytmp[i] = 0.0;

        // left border
        for (unsigned int i = 0; i < n_border; ++i) {
            float sum = 0.0;

            xtmp[0] = cur_line[(n_border - i) * stride];
            for (unsigned int j = 0; j < 4; ++j)
                sum += n_causal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            ytmp[0] = sum;
        }

        // causal pass
        for (unsigned int i = 0; i < n_pixels; ++i) {
            float sum = 0.0;

            xtmp[0] = cur_line[i * stride];
            for (unsigned int j = 0; j < 4; ++j)
                sum += n_causal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            tmpbfr[i] = sum;
            ytmp[0] = sum;
        }

        // reset variables for anti-causal pass
        for (unsigned int i = 0; i < 4; ++i)
            xtmp[i] = ytmp[i] = 0.0;

        // right border
        for (int i = n_border; i > 0; --i) {
            float sum = 0.0;

            for (unsigned int j = 0; j < 4; ++j)
                sum += n_anticausal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            xtmp[0] = cur_line[(n_pixels - i) * stride];
            ytmp[0] = sum;
        }

        // anti-causal pass
        for (int i = n_pixels - 1; i >= 0; --i) {
            float sum = 0.0;

            for (unsigned int j = 0; j < 4; ++j)
                sum += n_anticausal[j] * xtmp[j];
            for (unsigned int j = 0; j < 4; ++j)
                sum -= d[j] * ytmp[j];
            for (unsigned int j = 3; j > 0; --j) {
                xtmp[j] = xtmp[j - 1];
                ytmp[j] = ytmp[j - 1];
            }

            xtmp[0] = cur_line[i * stride];
            ytmp[0] = sum;
            cur_output[i * stride] = tmpbfr[i] + sum;
        }
    }
}

} // namespace detail

} // namespace fastfilters