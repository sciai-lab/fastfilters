#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fastfilters.hxx"

namespace py = pybind11;

struct IIRCoefficients
{
    double sigma;
    unsigned int order;
    std::array<float, 4> d, n_causal, n_anticausal;

    IIRCoefficients(const double sigma, const unsigned int order) : sigma(sigma), order(order)
    {
        fastfilters::deriche::compute_coefs(sigma, order, n_causal, n_anticausal, d);
    }

    std::string repr()
    {
        std::ostringstream oss;
        oss << "<pyfastfilters.IIRCoefficients with sigma = " << sigma << " and order = " << order << ">";

        return oss.str();
    }
};

static py::array_t<float> iir_filter(IIRCoefficients &coefs, py::array_t<float> input)
{
    py::buffer_info info_in = input.request();
    const unsigned int ndim = (unsigned int)info_in.ndim;

    if (info_in.ndim <= 0)
        throw std::runtime_error("Number of dimensions must be > 0");

    auto result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), ndim,
                                            info_in.shape, info_in.strides));
    py::buffer_info info_out = result.request();

    const float *inptr = (float *)info_in.ptr;
    float *outptr = (float *)info_out.ptr;

    unsigned int n_border = 0;

    unsigned int n_times = 1;
    for (unsigned int i = 1; i < ndim; ++i)
        n_times *= info_in.shape[i];

    // if (fastfilters::detail::cpu_has_avx2())
    //	fastfilters::iir::convolve_iir_inner_single_avx(inptr, info_in.shape[0], n_times, outptr, coefs.n_causal,
    // coefs.n_anticausal, coefs.d, n_border);
    // else
    fastfilters::iir::convolve_iir_inner_single(inptr, info_in.shape[0], n_times, outptr, coefs.n_causal,
                                                coefs.n_anticausal, coefs.d, n_border);

    for (unsigned int i = 1; i < ndim; ++i) {
        n_times = 1;
        for (unsigned int j = 0; j < ndim; ++j)
            if (j != i)
                n_times *= info_in.shape[j];

        std::cout << i << " " << n_times << " " << info_in.shape[i] << std::endl;
        // if (fastfilters::detail::cpu_has_avx2())
        //	fastfilters::iir::convolve_iir_outer_single_avx(inptr, info_in.shape[i], n_times, outptr, coefs.n_causal,
        // coefs.n_anticausal, coefs.d, n_border);
        // else
        fastfilters::iir::convolve_iir_outer_single(outptr, info_in.shape[i], n_times, outptr, coefs.n_causal,
                                                    coefs.n_anticausal, coefs.d, n_border, n_times);
    }

    return result;
}

PYBIND11_PLUGIN(pyfastfilters)
{
    py::module m("pyfastfilters", "fast gaussian kernel and derivative filters");

    py::class_<IIRCoefficients>(m, "IIRCoefficients")
        .def(py::init<const double, const unsigned int>())
        .def("__repr__", &IIRCoefficients::repr)
        .def_readonly("sigma", &IIRCoefficients::sigma)
        .def_readonly("order", &IIRCoefficients::order);

    m.def("iir_filter", &iir_filter, "apply IIR filter to all dimensions of array and return result.");

    return m.ptr();
}