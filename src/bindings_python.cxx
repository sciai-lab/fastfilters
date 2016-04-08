// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fastfilters.h"
#include "common.h"

#include <string>

namespace py = pybind11;

namespace
{

struct FIRKernel
{
    fastfilters_kernel_fir_t kernel;
    const unsigned order;
    const double sigma;

    FIRKernel(unsigned order, double sigma) : order(order), sigma(sigma)
    {
        kernel = fastfilters_kernel_fir_gaussian(order, sigma);

        if (!kernel)
            throw std::runtime_error("fastfilters_kernel_fir_gaussian returned NULL.");
    }

    ~FIRKernel()
    {
        fastfilters_kernel_fir_free(kernel);
    }

    std::string __repr__()
    {
        std::stringstream oss;
        oss << "<fastfilters.FIRKernel with sigma = " << sigma << " and order = " << order << ">";

        return oss.str();
    }
};

py::array_t<float> linalg_ev2d(py::array_t<float> &mtx)
{
    py::buffer_info info = mtx.request();

    if (info.ndim != 2)
        throw std::runtime_error("matrix must have two dimensions.");

    if (info.shape[0] != 3)
        throw std::runtime_error("1st dimension must have len = 3.");

    auto result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), 2,
                                            {2, info.shape[1]}, {sizeof(float) * info.shape[1], sizeof(float)}));
    py::buffer_info info_out = result.request();

    float *inptr = (float *)info.ptr;
    float *xx = inptr;
    float *xy = inptr + info.strides[0] / sizeof(float);
    float *yy = inptr + 2 * info.strides[0] / sizeof(float);

    float *outptr = (float *)info_out.ptr;
    float *ev_small = outptr;
    float *ev_big = outptr + info_out.strides[0] / sizeof(float);

    fastfilters_linalg_ev2d(xx, xy, yy, ev_small, ev_big, info.shape[1]);

    return result;
}
};

PYBIND11_PLUGIN(fastfilters)
{
    py::module m_fastfilters("fastfilters", "fast gaussian kernel and derivative filters");

    fastfilters_init(PyMem_Malloc, PyMem_Free);

    py::class_<FIRKernel>(m_fastfilters, "FIRKernel")
        .def(py::init<unsigned, double>())
        .def_readonly("sigma", &FIRKernel::sigma)
        .def_readonly("order", &FIRKernel::order);

    m_fastfilters.def("linalg_ev2d", &linalg_ev2d);

    return m_fastfilters.ptr();
}