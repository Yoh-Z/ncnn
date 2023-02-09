// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cmpop.h"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include <math.h>

namespace ncnn {

CmpOp::CmpOp()
{
    one_blob_only = true;
    support_inplace = false;
}

int CmpOp::load_param(const ParamDict& pd)
{
    value = pd.get(0, 0);
    cmp_operation = pd.get(7, 0);

    return 0;
}

int CmpOp::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    top_blob.create_like(bottom_blob);

    if (cmp_operation == 2)
    {
        __m256 _q = _mm256_set1_ps(value);
        __m256 _cmp = _mm256_set1_ps(0x1);
        for (int q = 0 ; q < channels; q ++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* cmp_ptr = top_blob.channel(q);
            for (int ii = 0; ii < w * h * d; ii ++)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_cmp_ps(_p, _q, _CMP_GT_OS);
                _p = _mm256_and_ps(_p, _cmp);
                _mm256_storeu_ps(cmp_ptr, _p);
                if (*cmp_ptr > 1.1f || *cmp_ptr < -0.1f) printf("%f ", *cmp_ptr);
            }
        }
    }

    return 0;
}

int CmpOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    Mat& top_blob = top_blobs[0];

    

    return 0;
}

} // namespace ncnn
