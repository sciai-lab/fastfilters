#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fastfilters.hxx"

namespace py = pybind11;

static py::array_t<float> iir_filter(fastfilters::iir::Coefficients &coefs, py::array_t<float> input)
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

    unsigned int n_times = 1;
    for (unsigned int i = 0; i < ndim - 1; ++i)
        n_times *= info_in.shape[i];

    fastfilters::iir::convolve_iir_inner_single(inptr, info_in.shape[ndim - 1], n_times, outptr, coefs);

    for (unsigned int i = 0; i < ndim - 1; ++i) {
        n_times = 1;
        for (unsigned int j = 0; j < ndim; ++j)
            if (j != i)
                n_times *= info_in.shape[j];

        fastfilters::iir::convolve_iir_outer_single(outptr, info_in.shape[i], n_times, outptr, coefs);
    }

    return result;
}

PYBIND11_PLUGIN(fastfilters)
{
    py::module m("fastfilters", "fast gaussian kernel and derivative filters");

    py::class_<fastfilters::iir::Coefficients>(m, "IIRCoefficients")
        .def(py::init<const double, const unsigned int>())
        .def("__repr__", [](const fastfilters::iir::Coefficients &a) {
            std::ostringstream oss;
            oss << "<fastfilters.IIRCoefficients with sigma = " << a.sigma << " and order = " << a.order << ">";

            return oss.str();
        })
        .def_readonly("sigma", &fastfilters::iir::Coefficients::sigma)
        .def_readonly("order", &fastfilters::iir::Coefficients::order);

    py::class_<fastfilters::fir::Kernel>(m, "FIRKernel")
        .def(py::init<const bool, const std::vector<float>>())
        .def("__repr__", [](const fastfilters::fir::Kernel &a) {
            std::ostringstream oss;
            oss << "<fastfilters.FIRKernel with symmetric = " << a.is_symmetric << " and length = " << a.len() << ">";

            return oss.str();
        })
        .def("__getitem__", [](const fastfilters::fir::Kernel &s, size_t i) {
            if (i >= s.len())
                throw py::index_error();
            return s[i];
        });

    m.def("iir_filter", &iir_filter, "apply IIR filter to all dimensions of array and return result.");

    m.def("cpu_has_avx2", &fastfilters::detail::cpu_has_avx2);

    return m.ptr();
}