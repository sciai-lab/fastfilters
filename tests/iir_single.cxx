#include "fastfilters.hxx"

#include <string.h>

int main()
{
	float n_causal[4], n_anticausal[4], d[4];
	float input[2*512];
	float output[2*512];

	memset(input, 0, sizeof(input));

	fastfilters::deriche::compute_coefs(5.0, 0, n_causal, n_anticausal, d);
	fastfilters::detail::convolve_iir_inner_single(
		input,
		512, 2,
		output,
		n_causal, n_anticausal, d,
		32);

	return 0;
}