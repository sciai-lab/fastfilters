
master: [![Build Status](https://travis-ci.org/svenpeter42/fastfilters.svg?branch=master)](https://travis-ci.org/svenpeter42/fastfilters) [![Build status](https://ci.appveyor.com/api/projects/status/obc03rs0cwisnsdv/branch/master?svg=true)](https://ci.appveyor.com/project/svenpeter42/fastfilters/branch/master)

devel: [![Build Status](https://travis-ci.org/svenpeter42/fastfilters.svg?branch=devel)](https://travis-ci.org/svenpeter42/fastfilters) [![Build status](https://ci.appveyor.com/api/projects/status/obc03rs0cwisnsdv/branch/master?svg=true)](https://ci.appveyor.com/project/svenpeter42/fastfilters/branch/devel)

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

	% conda install -c ilastik fastfilters


Gentoo Installation (development)
------------

	% git clone https://github.com/svenpeter42/fastfilters.git
	% cd fastfilters/pkg/gentoo/sci-libs/fastfilters
	% sudo ebuild fastfilters-9999.ebuild manifest clean merge
