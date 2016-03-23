#ifndef FASTFILTERS_VECTOR_HXX
#define FASTFILTERS_VECTOR_HXX

// faster than std::vector because of missing out-of-bounds safety checks
template <typename T> class ConstantVector
{
  public:
    inline ConstantVector(const unsigned int size)
    {
        ptr = new T[size];
    }

    inline ~ConstantVector()
    {
        delete ptr;
    }

    inline const T &operator[](unsigned int i) const
    {
        return ptr[i];
    }

    inline T &operator[](unsigned int i)
    {
        return ptr[i];
    }

    inline T *operator+(unsigned int i)
    {
        return ptr + i;
    }

  private:
    T *ptr;
};

#ifdef __AVX2__
#include <immintrin.h>

class AVXVector
{
    class Proxy
    {
      private:
        float *ptr;

      public:
        inline Proxy(float *ptr) : ptr(ptr)
        {
        }

        inline void operator=(__m256 v)
        {
            _mm256_store_ps(ptr, v);
        }

        inline operator __m256()
        {
            return _mm256_load_ps(ptr);
        }
    };

    inline AVXVector(const unsigned int size)
    {
        int res = posix_memalign((void **)&ptr, 32, sizeof(float) * size * 8);

        if (res < 0 || ptr == NULL)
            throw std::runtime_error("posix_memalign failed.");
    }

    inline ~AVXVector()
    {
        free(ptr);
    }

    inline const __m256 operator[](unsigned int i) const
    {
        return _mm256_load_ps(ptr + 8 * i);
    }

    inline Proxy operator[](unsigned int i)
    {
        return Proxy(ptr + 8 * i);
    }

  private:
    float *ptr;
};
#endif

#endif