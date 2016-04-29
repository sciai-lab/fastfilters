# Copyright 1999-2014 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2
# $Id$

EAPI=6
PYTHON_COMPAT=( python2_7 python3_4 )
PYTHON_REQ_USE="threads,xml"
inherit python-r1 git-r3 waf-utils

DESCRIPTION="fast gaussian and derivative convolutional filters"
HOMEPAGE="https://github.com/svenpeter42/fastfilters"
EGIT_REPO_URI="https://github.com/svenpeter42/fastfilters.git"
EGIT_BRANCH="devel"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64"
IUSE=""

RDEPEND="
	${PYTHON_DEPS}
"

DEPEND="${RDEPEND}"

REQUIRED_USE=""

DOCS=( README.md )

pkg_setup() {
	python_setup
}

src_configure() {
	fastfilters_configure() {
		CFLAGS="" CXXFLAGS="" ./waf configure --prefix=${D} --python=${PYTHON} -o ${BUILD_DIR} --pythondir=${D}/${PORTAGE_PYTHONPATH} --pythonarchdir=${D}/${PORTAGE_PYTHONPATH}
	}
	python_setup
	python_foreach_impl fastfilters_configure
}

src_compile() {
	fastfilters_compile() {
		./waf -v
	}
	python_foreach_impl fastfilters_compile
}

src_install() {
	fastfilters_install() {
		./waf install
	}
	python_foreach_impl fastfilters_install
}
