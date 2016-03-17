#ifndef FASTFILTERS_HXX
#define FASTFILTERS_HXX

namespace fastfilters {

namespace detail {

bool cpu_has_avx2();

}


template<unsigned N>
class FastFilterArrayView
{
public:
	const float *baseptr;
	const unsigned int n_pixels[N];
	const unsigned int n_channels;
};

}

#endif