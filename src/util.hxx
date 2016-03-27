#ifndef UTIL_HXX
#define UTIL_HXX 1

#include <stdint.h>
#include "config.h"

namespace fastfilters
{

namespace detail
{

static inline void *avx_memalign(std::size_t len)
{

#ifdef HAVE_POSIX_MEMALIGN
    void *ptr = NULL;
    int res = posix_memalign(&ptr, 32, len);

    if (res < 0 || ptr == NULL)
        throw std::runtime_error("posix_memalign failed.");

    return ptr;
#elif defined(HAVE_ALIGNED_MALLOC)
    void *ptr = _aligned_malloc(len, 32);

    if (ptr == NULL)
        throw std::runtime_error("_aligned_malloc failed.");

    return ptr;
#else
#error "No known way to allocate aligned memory."
#endif
}

template <typename T> static inline void avx_free(T *ptr)
{
#ifdef HAVE_POSIX_MEMALIGN
    free(ptr);
#elif defined(HAVE_ALIGNED_MALLOC)
    _aligned_free(ptr);
#else
#error "No known way to free aligned memory."
#endif
}
}
}

#endif