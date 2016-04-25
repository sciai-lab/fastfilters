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

    unsigned int len()
    {
        return fastfilters_kernel_fir_get_length(kernel);
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
               np_info.strides[ff_ndim] == sizeof(float)) {
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

py::array_t<float> convolve_2d_fir(py::array_t<float> &input, FIRKernel *k0, FIRKernel *k1, unsigned x0 = 0,
                                   unsigned y0 = 0, unsigned x1 = 0, unsigned y1 = 0)
{
    fastfilters_array2d_t ff;
    fastfilters_array2d_t ff_out;

    py::array_t<float> result;

    if (x0 == 0 && x1 == 0 && y0 == 0 && y1 == 0)
        result = array_like(input);
    else {
        py::buffer_info input_info = input.request();
        size_t n_x = x1 - x0;
        size_t n_y = y1 - y0;

        if (input_info.ndim == 2)
            result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), 2,
                                               {n_y, n_x}, {sizeof(float) * n_x, sizeof(float)}));
        else if (input_info.ndim == 3)
            result = py::array(py::buffer_info(
                nullptr, sizeof(float), py::format_descriptor<float>::value(), 2, {n_y, n_x, input_info.shape[2]},
                {sizeof(float) * n_x * input_info.strides[2], input_info.strides[2], input_info.strides[1]}));
        else
            throw std::logic_error("Not implemented.");
    }

    convert_py2ff(input, ff);
    convert_py2ff(result, ff_out);

    if (!fastfilters_fir_convolve2d(&ff, k0->kernel, k1->kernel, &ff_out, x0, y0, x1, y1))
        throw std::logic_error("fastfilters_fir_convolve2d returned false.");

    return result;
}

py::array_t<float> convolve_3d_fir(py::array_t<float> &input, FIRKernel *k0, FIRKernel *k1, FIRKernel *k2,
                                   unsigned x0 = 0, unsigned y0 = 0, unsigned z0 = 0, unsigned x1 = 0, unsigned y1 = 0,
                                   unsigned z1 = 0)
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

py::array_t<float> convolve_fir_roi2d(py::array_t<float> &input, std::vector<FIRKernel *> k,
                                      std::pair<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned>> roi)
{
    if (k.size() != 2)
        throw std::logic_error("Kernel number (2) and ROI shape mismatch.");

    return convolve_2d_fir(input, k[0], k[1], roi.first.first, roi.first.second, roi.second.first, roi.second.second);
}

py::array_t<float>
convolve_fir_roi3d(py::array_t<float> &input, std::vector<FIRKernel *> k,
                   std::pair<std::tuple<unsigned, unsigned, unsigned>, std::tuple<unsigned, unsigned, unsigned>> roi)
{
    if (k.size() != 3)
        throw std::logic_error("Kernel number (3) and ROI shape mismatch.");

    return convolve_3d_fir(input, k[0], k[1], k[2], std::get<0>(roi.first), std::get<1>(roi.first),
                           std::get<2>(roi.first), std::get<0>(roi.second), std::get<1>(roi.second),
                           std::get<2>(roi.second));
}

struct ConvolveGaussian {
    unsigned order;
    double sigma;

    ConvolveGaussian(unsigned order, double sigma) : order(order), sigma(sigma)
    {
    }

    bool operator()(fastfilters_array2d_t &in, fastfilters_array2d_t &out)
    {
        return fastfilters_fir_gaussian2d(&in, order, sigma, &out);
    }
};

template <typename ConvolveFunctor> py::array_t<float> filter2d_binding(py::array_t<float> &input, ConvolveFunctor &fn)
{
    fastfilters_array2d_t ff;
    fastfilters_array2d_t ff_out;

    auto result = array_like(input);
    convert_py2ff(input, ff);
    convert_py2ff(result, ff_out);

    if (!fn(ff, ff_out))
        throw std::logic_error("convolution failed.");

    return result;
}

py::array_t<float> gaussian2d(py::array_t<float> &input, unsigned order, double sigma)
{
    ConvolveGaussian fn(order, sigma);
    return filter2d_binding(input, fn);
}

struct ConvolveGradMag {
    double sigma;

    ConvolveGradMag(double sigma) : sigma(sigma)
    {
    }

    bool operator()(fastfilters_array2d_t &in, fastfilters_array2d_t &out)
    {
        return fastfilters_fir_gradmag2d(&in, sigma, &out);
    }
};

py::array_t<float> gradmag2d(py::array_t<float> &input, double sigma)
{
    ConvolveGradMag fn(sigma);
    return filter2d_binding(input, fn);
}

struct ConvolveLaPlacian {
    double sigma;

    ConvolveLaPlacian(double sigma) : sigma(sigma)
    {
    }

    bool operator()(fastfilters_array2d_t &in, fastfilters_array2d_t &out)
    {
        return fastfilters_fir_laplacian2d(&in, sigma, &out);
    }
};

py::array_t<float> laplacian2d(py::array_t<float> &input, double sigma)
{
    ConvolveLaPlacian fn(sigma);
    return filter2d_binding(input, fn);
}

template <class ConvolveFunctor> py::array_t<float> filter_ev_2d_binding(py::array_t<float> &input, ConvolveFunctor &fn)
{
    fastfilters_array2d_t ff;
    fastfilters_array2d_t ff_out_xx, ff_out_yy, ff_out_xy;

    auto out_xx = array_like(input);
    auto out_yy = array_like(input);
    auto out_xy = array_like(input);

    convert_py2ff(input, ff);

    convert_py2ff(out_xx, ff_out_xx);
    convert_py2ff(out_yy, ff_out_yy);
    convert_py2ff(out_xy, ff_out_xy);

    if (!fn(ff, ff_out_xx, ff_out_xy, ff_out_yy))
        throw std::logic_error("convolution failed.");

    const size_t n_pixels = ff.n_x * ff.n_y * ff.n_channels;

    auto result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), 2,
                                            {2, n_pixels}, {sizeof(float) * n_pixels, sizeof(float)}));
    py::buffer_info info_out = result.request();

    float *xx = ff_out_xx.ptr;
    float *xy = ff_out_xy.ptr;
    float *yy = ff_out_yy.ptr;

    float *outptr = (float *)info_out.ptr;
    float *ev_small = outptr;
    float *ev_big = outptr + info_out.strides[0] / sizeof(float);

    fastfilters_linalg_ev2d(xx, xy, yy, ev_small, ev_big, n_pixels);

    return result;
}

struct ConvolveHessian {
    double sigma;

    ConvolveHessian(double sigma) : sigma(sigma)
    {
    }

    bool operator()(fastfilters_array2d_t &in, fastfilters_array2d_t &xx, fastfilters_array2d_t &xy,
                    fastfilters_array2d_t &yy)
    {
        return fastfilters_fir_hog2d(&in, sigma, &xx, &xy, &yy);
    }
};

py::array_t<float> hog2d(py::array_t<float> &input, double sigma)
{
    ConvolveHessian fn(sigma);
    return filter_ev_2d_binding(input, fn);
}

struct ConvolveST {
    double sigma_inner, sigma_outer;

    ConvolveST(double sigma_inner, double sigma_outer) : sigma_inner(sigma_inner), sigma_outer(sigma_outer)
    {
    }

    bool operator()(fastfilters_array2d_t &in, fastfilters_array2d_t &xx, fastfilters_array2d_t &xy,
                    fastfilters_array2d_t &yy)
    {
        return fastfilters_fir_structure_tensor(&in, sigma_inner, sigma_outer, &xx, &xy, &yy);
    }
};

py::array_t<float> st2d(py::array_t<float> &input, double sigma_inner, double sigma_outer)
{
    ConvolveST fn(sigma_inner, sigma_outer);
    return filter_ev_2d_binding(input, fn);
}
};

PYBIND11_PLUGIN(fastfilters)
{
    py::module m_fastfilters("fastfilters", "fast gaussian kernel and derivative filters");

    fastfilters_init_ex(PyMem_Malloc, PyMem_Free);

    py::class_<FIRKernel>(m_fastfilters, "FIRKernel")
        .def(py::init<unsigned, double>())
        .def("len", &FIRKernel::len)
        .def_readonly("sigma", &FIRKernel::sigma)
        .def_readonly("order", &FIRKernel::order);

    m_fastfilters.def("linalg_ev2d", &linalg_ev2d);
    m_fastfilters.def("convolve_fir", &convolve_fir, py::arg("input"), py::arg("kernels"));
    m_fastfilters.def("convolve_fir", &convolve_fir_roi2d, py::arg("input"), py::arg("kernels"), py::arg("roi"));
    m_fastfilters.def("convolve_fir", &convolve_fir_roi3d, py::arg("input"), py::arg("kernels"), py::arg("roi"));

    m_fastfilters.def("gaussian2d", &gaussian2d);
    m_fastfilters.def("gradmag2d", &gradmag2d);
    m_fastfilters.def("laplacian2d", &laplacian2d);
    m_fastfilters.def("hog2d", &hog2d);
    m_fastfilters.def("st2d", &st2d);

    return m_fastfilters.ptr();
}
