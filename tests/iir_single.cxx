#include "fastfilters.hxx"

int main()
{
	float n_causal[4], n_anticausal[4], d[4];

	fastfilters::deriche::compute_coefs(5.0, 0, n_causal, n_anticausal, d);

	return 0;
}