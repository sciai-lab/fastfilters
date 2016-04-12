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

// based on vigra's include/gaussian.gxx with the following license:
/************************************************************************/
/*                                                                      */
/*               Copyright 1998-2004 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include "fastfilters.h"
#include "common.h"

fastfilters_kernel_fir_t fastfilters_kernel_fir_gaussian(unsigned int order, double sigma)
{
    double norm;
    double sigma2 = -0.5 / sigma / sigma;

    if (order > 2)
        return NULL;

    if (sigma < 0)
        return NULL;

    fastfilters_kernel_fir_t kernel = fastfilters_memory_alloc(sizeof(struct _fastfilters_kernel_fir_t));
    if (!kernel)
        return NULL;

    kernel->len = ceil((3.0 + 0.5 * order) * sigma);
    kernel->coefs = fastfilters_memory_alloc(sizeof(float) * (kernel->len + 1));
    if (!kernel->coefs) {
        fastfilters_memory_free(kernel);
        return NULL;
    }

    if (order == 1)
        kernel->is_symmetric = false;
    else
        kernel->is_symmetric = true;

    switch (order) {
    case 1:
    case 2:
        norm = -1.0 / (sqrt(2.0 * M_PI) * pow(sigma, 3));
        break;

    default:
        norm = 1.0 / (sqrt(2.0 * M_PI) * sigma);
        break;
    }

    for (unsigned int x = 0; x <= kernel->len; ++x) {
        double x2 = x * x;
        double g = norm * exp(x2 * sigma2);
        switch (order) {
        case 0:
            kernel->coefs[x] = g;
            break;
        case 1:
            kernel->coefs[x] = x * g;
        case 2:
            kernel->coefs[x] = (1.0 - (x / sigma) * (x / sigma)) * g;
        }
    }

    return kernel;
}

void fastfilters_kernel_fir_free(fastfilters_kernel_fir_t kernel)
{
    fastfilters_memory_free(kernel->coefs);
    fastfilters_memory_free(kernel);
}
