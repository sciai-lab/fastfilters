#include "fastfilters.hxx"
#include <iostream>
namespace fastfilters
{

namespace fir
{

// faster than std::vector because of missing out-of-bounds safety checks
template <typename T> class ConstantVector
{
  public:
    inline ConstantVector(const unsigned int size)
    {
        ptr = new T[size];
    }

    ~ConstantVector()
    {
        delete ptr;
    }

    const T &operator[](unsigned int i) const
    {
        return ptr[i];
    }

    T &operator[](unsigned int i)
    {
        return ptr[i];
    }

  private:
    T *ptr;
};

void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     const unsigned int dim_stride, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();
    ConstantVector<float> tmpline(n_pixels);

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * dim_stride;
        float *cur_output = output + i * dim_stride;

        for (unsigned int j = 0; j < n_pixels; ++j)
            tmpline[j] = cur_input[j];

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + (int)j < 0)
                    sum += kernel[k] * tmpline[j - kreal];
                else
                    sum += kernel[k] * tmpline[j + kreal];
            }

            cur_output[j] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * tmpline[j + kreal];
            }

            cur_output[j] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + j >= n_pixels)
                    sum += kernel[k] * tmpline[j - kreal];
                else
                    sum += kernel[k] * tmpline[j + kreal];
            }

            cur_output[j] = sum;
        }
    }
}

template <typename T> class FinalVector
{
  public:
    FinalVector(unsigned int size)
    {
        v.reserve(size);
    }
    const T &at(unsigned int i) const
    {
        return v.at(i);
    }
    T &at(unsigned int i)
    {
        return v.at(i);
    }
    T &operator[](unsigned int i)
    {
        return at(i);
    }
    const T &operator[](unsigned int i) const
    {
        return at(i);
    }
    void push_back(const T &x);
    size_t size() const
    {
        return v.size();
    }
    size_t capacity() const
    {
        return v.size();
    }

  private:
    std::vector<T> v;
};

void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                                     const unsigned n_times, const unsigned dim_stride, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();
    ConstantVector<float> tmpline(n_pixels);

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * dim_stride;
        float *cur_output = output + i * dim_stride;

        for (unsigned int j = 0; j < n_pixels; ++j)
            tmpline[j] = cur_input[j * pixel_stride];

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + (int)j < 0)
                    sum += kernel[k] * tmpline[(j - kreal)];
                else
                    sum += kernel[k] * tmpline[(j + kreal)];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * tmpline[(j + kreal)];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + j >= n_pixels)
                    sum += kernel[k] * tmpline[(j - kreal)];
                else
                    sum += kernel[k] * tmpline[(j + kreal)];
            }

            cur_output[j * pixel_stride] = sum;
        }
    }
}

} // namespace detail

} // namespace fastfilters