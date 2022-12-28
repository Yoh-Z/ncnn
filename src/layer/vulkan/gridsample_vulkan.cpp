// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gridsample_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

GridSample_vulkan::GridSample_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_gridsample = 0;
    pipeline_gridsample_pack4 = 0;
    pipeline_gridsample_pack8 = 0;
}

int GridSample_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1 || shape.dims == 2)
        return -100;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4
                                                                                                                       : 1;
    int out_elempack = 1;
    if (out_shape.dims == 1 || out_shape.dims == 2)
        return -100;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4
                                                                                                                                           : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(3 + 11);
    specializations[0].i = sample_type;
    specializations[1].i = padding_mode;
    specializations[2].i = align_corner;
    specializations[3 + 0].i = shape_packed.dims;
    specializations[3 + 1].i = shape_packed.w;
    specializations[3 + 2].i = shape_packed.h;
    specializations[3 + 3].i = shape_packed.d;
    specializations[3 + 4].i = shape_packed.c;
    specializations[3 + 5].i = shape_packed.cstep;
    specializations[3 + 6].i = out_shape_packed.dims;
    specializations[3 + 7].i = out_shape_packed.w;
    specializations[3 + 8].i = out_shape_packed.h;
    specializations[3 + 9].i = out_shape_packed.d;
    specializations[3 + 10].i = out_shape_packed.c;
    specializations[3 + 11].i = out_shape_packed.cstep;

    Mat local_size_xyz;
    if (out_shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }
    if (out_shape_packed.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h * out_shape_packed.d);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_gridsample = new Pipeline(vkdev);
        pipeline_gridsample->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample->create(LayerShaderType::gridsample, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_gridsample_pack4 = new Pipeline(vkdev);
        pipeline_gridsample_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_pack4->create(LayerShaderType::gridsample_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_gridsample_pack8 = new Pipeline(vkdev);
        pipeline_gridsample_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_pack8->create(LayerShaderType::gridsample_pack8, opt, specializations);
    }

    return 0;
}

int GridSample_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gridsample;
    pipeline_gridsample = 0;

    delete pipeline_gridsample_pack4;
    pipeline_gridsample_pack4 = 0;

    delete pipeline_gridsample_pack8;
    pipeline_gridsample_pack8 = 0;

    return 0;
}

int GridSample_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_blobs[0].elempack;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blobs[0];
    bindings[1] = top_blobs[0];
    bindings[2] = bottom_blobs[1];

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_blobs[0].dims;
    constants[1].i = bottom_blobs[0].w;
    constants[2].i = bottom_blobs[0].h;
    constants[3].i = bottom_blobs[0].d;
    constants[4].i = bottom_blobs[0].c;
    constants[5].i = 0; //bottom_blob cstep
    constants[0].i = top_blobs[0].dims;
    constants[1].i = top_blobs[0].w;
    constants[2].i = top_blobs[0].h;
    constants[3].i = top_blobs[0].d;
    constants[4].i = top_blobs[0].c;
    constants[5].i = 0; //top_blob cstep

    const Pipeline* pipeline = elempack == 8 ? pipeline_gridsample_pack8
                               : elempack == 4 ? pipeline_gridsample_pack4
                               : pipeline_gridsample;

    cmd.record_pipeline(pipeline, bindings, constants, top_blobs[0]);

    return 0;
}


int GridSample_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_blobs[0].elempack;

    std::vector<VkImageMat> bindings(3);
    bindings[0] = bottom_blobs[0];
    bindings[1] = top_blobs[0];
    bindings[2] = bottom_blobs[1];

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_blobs[0].dims;
    constants[1].i = bottom_blobs[0].w;
    constants[2].i = bottom_blobs[0].h;
    constants[3].i = bottom_blobs[0].d;
    constants[4].i = bottom_blobs[0].c;
    constants[5].i = 0; //bottom_blob cstep
    constants[0].i = top_blobs[0].dims;
    constants[1].i = top_blobs[0].w;
    constants[2].i = top_blobs[0].h;
    constants[3].i = top_blobs[0].d;
    constants[4].i = top_blobs[0].c;
    constants[5].i = 0; //top_blob cstep

    const Pipeline* pipeline = elempack == 8 ? pipeline_gridsample_pack8
                               : elempack == 4 ? pipeline_gridsample_pack4
                               : pipeline_gridsample;

    cmd.record_pipeline(pipeline, bindings, constants, top_blobs[0]);

    return 0;
}

} // namespace ncnn
