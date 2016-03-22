#ifndef FASTFILTERS_VIGRA_HXX
#define FASTFILTERS_VIGRA_HXX

#include "fastfilters.hxx"
#include <vigra/multi_array.hxx>

namespace fastfilters
{

template <unsigned ndim>
inline void separableConvolveMultiArray(const vigra::MultiArrayView<ndim, float> input,
                                        vigra::MultiArrayView<ndim, float> &output, fir::Kernel &kernel)
{
    const float *inptr = input.data();
    float *outptr = output.data();

    if (ndim == 1) {
        fastfilters::fir::convolve_fir_inner_single(inptr, input.size(), 1, input.size(), outptr, kernel);
        return;
    }

    unsigned int n_times = 1;
    for (unsigned int i = 0; i < ndim - 1; ++i)
        n_times *= input.shape()[i];
    fastfilters::fir::convolve_fir_inner_single(inptr, input.shape()[ndim - 1], n_times, input.shape()[ndim - 1],
                                                outptr, kernel);

    for (int i = ndim - 2; i >= 0; --i) {
        unsigned int pixel_stride = 1;
        n_times = 1;
        for (int j = 0; j < ndim; ++j) {
            if (j != i)
                n_times *= input.shape()[j];
            if (j > i)
                pixel_stride *= input.shape()[j];
        }

        fastfilters::fir::convolve_fir(outptr, input.shape()[i], pixel_stride, n_times, 1, outptr, kernel);
    }
}

} // namespace fastfilters
#endif