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

	FastFilterArrayView(const float *baseptr, const unsigned int n_pixels[N], const unsigned int n_channels) : baseptr(baseptr), n_pixels(n_pixels), n_channels(n_channels)
	{

	}
};

namespace iir
{
	void convolve_iir_inner_single(
		const float *input,
		const unsigned int n_pixels, const unsigned n_times,
		float *output,
		const float *n_causal, const float *n_anticausal, const float *d,
		const unsigned n_border);

	void convolve_iir_outer_single(
		const float *input,
		const unsigned int n_pixels, const unsigned n_times,
		float *output,
		const float *n_causal, const float *n_anticausal, const float *d,
		const unsigned n_border,
		const unsigned stride);

	void convolve_iir_inner_single_avx(
		const float *input,
		const unsigned int n_pixels, const unsigned n_times,
		float *output,
		const float *n_causal, const float *n_anticausal, const float *d,
		const unsigned n_border);

	void convolve_iir_outer_single_avx(
		const float *input,
		const unsigned int n_pixels, const unsigned n_times,
		float *output,
		const float *n_causal, const float *n_anticausal, const float *d,
		const unsigned n_border);
}

namespace detail
{
	template<unsigned N>
	void gaussian_fir_inner(FastFilterArrayView<N> &input, float *output, const unsigned order, const double sigma)
	{
		(void)input;
		(void)output;
		(void)sigma;
	}

	template<unsigned N>
	void gaussian_fir_outer(FastFilterArrayView<N> &input, const unsigned n_dim, float *output, const unsigned order, const double sigma)
	{
		(void)input;
		(void)output;
		(void)sigma;
		(void)n_dim;
	}

	template<unsigned N>
	void gaussian_iir_inner(FastFilterArrayView<N> &input, float *output, const unsigned order, const double sigma)
	{
		(void)input;
		(void)output;
		(void)sigma;
	}

	template<unsigned N>
	void gaussian_iir_outer(FastFilterArrayView<N> &input, const unsigned n_dim, float *output, const unsigned order, const double sigma)
	{
		(void)input;
		(void)output;
		(void)sigma;
		(void)n_dim;
	}

	template<unsigned N>
	void gaussian_inner(FastFilterArrayView<N> &input, float *output, const unsigned order, const double sigma)
	{
		if (sigma < 3)
			gaussian_fir_inner(input, output, order, sigma);
		else
			gaussian_iir_inner(input, output, order, sigma);
	}

	template<unsigned N>
	void gaussian_outer(FastFilterArrayView<N> &input, const unsigned n_dim, float *output, const unsigned order, const double sigma)
	{
		if (sigma < 3)
			gaussian_fir_outer(input, n_dim, output, order, sigma);
		else
			gaussian_iir_outer(input, n_dim, output, order, sigma);
	}
} // namespace detail

namespace deriche
{
	void compute_coefs(const double sigma, const unsigned order, float *n_causal, float *n_anticausal, float *d);
}

template<unsigned N>
void gaussian(FastFilterArrayView<N> &input, float *output, const unsigned order, const double sigma)
{
	detail::gaussian_inner(input, output, order, sigma);

	for (unsigned int i = 1; i < N; ++i)
		detail::gaussian_outer(input, i, output, order, sigma);
}

} // namespace fastfilters

#endif