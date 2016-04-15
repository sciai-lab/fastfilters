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
#include <stdlib.h>

namespace py = pybind11;

namespace
{

struct FIRKernel {
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
        kernel = NULL;
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

template <typename fastfilters_array_t> struct ff_ndim_t {
};

template <> struct ff_ndim_t<fastfilters_array2d_t> {
    static const unsigned int ndim = 2;

    static void set_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
    {
    }
    static void set_stride_z(size_t /*n_z*/, fastfilters_array2d_t /*&k*/)
    {
    }
};

template <> struct ff_ndim_t<fastfilters_array3d_t> {
    static const unsigned int ndim = 3;

    static void set_z(size_t n_z, fastfilters_array3d_t &k)
    {
        k.n_z = n_z;
    }
    static void set_stride_z(size_t stride_z, fastfilters_array3d_t &k)
    {
        k.stride_z = stride_z;
    }
};

template <typename fastfilters_array_t> void convert_py2ff(py::array_t<float> &np, fastfilters_array_t &ff)
{
    const unsigned int ff_ndim = ff_ndim_t<fastfilters_array_t>::ndim;
    py::buffer_info np_info = np.request();

    if (np_info.ndim >= (int)ff_ndim) {
        ff.ptr = (float *)np_info.ptr;

        ff.n_x = np_info.shape[ff_ndim - 1];
        ff.stride_x = np_info.strides[ff_ndim - 1] / sizeof(float);

        ff.n_y = np_info.shape[ff_ndim - 2];
        ff.stride_y = np_info.strides[ff_ndim - 2] / sizeof(float);

        if (ff_ndim == 3) {
            ff_ndim_t<fastfilters_array_t>::set_z(np_info.shape[ff_ndim - 3], ff);
            ff_ndim_t<fastfilters_array_t>::set_stride_z(np_info.strides[ff_ndim - 3] / sizeof(float), ff);
        }
    } else {
        throw std::logic_error("Too few dimensions.");
    }

    if (np_info.ndim == ff_ndim) {
        ff.n_channels = 1;
    } else if ((np_info.ndim == ff_ndim + 1) && np_info.shape[ff_ndim] < 8 &&
               np_info.strides[ff_ndim + 1] == sizeof(float)) {
        ff.n_channels = np_info.shape[ff_ndim];
    } else {
        throw std::logic_error("Invalid number of dimensions or too many channels or stride between channels.");
    }
}

py::array_t<float> array_like(py::array_t<float> &base)
{
    py::buffer_info info = base.request();
    auto result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), info.ndim,
                                            info.shape, info.strides));

    return result;
}

py::array_t<float> convolve_2d_fir(py::array_t<float> &input, FIRKernel *k0, FIRKernel *k1)
{
    fastfilters_array2d_t ff;
    fastfilters_array2d_t ff_out;

    auto result = array_like(input);

    convert_py2ff(input, ff);
    convert_py2ff(result, ff_out);

    if (!fastfilters_fir_convolve2d(&ff, k0->kernel, k1->kernel, &ff_out, 0, 0, 0, 0))
        throw std::logic_error("fastfilters_fir_convolve2d returned false.");

    return result;
}

py::array_t<float> convolve_3d_fir(py::array_t<float> &input, FIRKernel *k0, FIRKernel *k1, FIRKernel *k2)
{
    fastfilters_array3d_t ff;
    convert_py2ff(input, ff);
    throw std::logic_error("3D unsupported.");
}

py::array_t<float> convolve_fir(py::array_t<float> &input, std::vector<FIRKernel *> k)
{
    if (k.size() == 2)
        return convolve_2d_fir(input, k[0], k[1]);
    else if (k.size() == 3)
        return convolve_3d_fir(input, k[0], k[1], k[2]);
    else
        throw std::logic_error("Invalid number of dimensions.");
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
    m_fastfilters.def("convolve_fir", &convolve_fir);

    return m_fastfilters.ptr();
}