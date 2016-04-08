// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "fastfilters.h"
#include "common.h"

#include <immintrin.h>

typedef void (*ev2d_fn_t)(const float *, const float *, const float *, float *, float *, const size_t);


void _ev2d_avx(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big, const size_t len);

static void _ev2d_default(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big, const size_t len)
{

}

static ev2d_fn_t g_ev2d_fn = NULL;


void fastfilters_linalg_init()
{
	if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX))
		g_ev2d_fn = _ev2d_avx;
	else
		g_ev2d_fn = _ev2d_default;
}

void fastfilters_linalg_ev2d(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big, const size_t len)
{
	g_ev2d_fn(xx, xy, yy, ev_small, ev_big, len);
}