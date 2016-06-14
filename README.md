
master: [![Build Status](https://travis-ci.org/svenpeter42/fastfilters.svg?branch=master)](https://travis-ci.org/svenpeter42/fastfilters)

devel: [![Build Status](https://travis-ci.org/svenpeter42/fastfilters.svg?branch=devel)](https://travis-ci.org/svenpeter42/fastfilters)

Installation (stable)
------------

	% git clone https://github.com/svenpeter42/fastfilters.git
	% cd fastfilters
        % mkdir build
        % cmake ..
        % make
        % make install


Conda Installation (stable)
------------

	% conda install -c svenpeter fastfilters


Gentoo Installation (development)
------------

	% git clone https://github.com/svenpeter42/fastfilters.git
	% cd fastfilters/pkg/gentoo/sci-libs/fastfilters
	% sudo ebuild fastfilters-9999.ebuild manifest clean merge
