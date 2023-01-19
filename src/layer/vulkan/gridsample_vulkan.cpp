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
#include "layer_type.h"

namespace ncnn {

GridSample_vulkan::GridSample_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_gridsample = 0;
    pipeline_gridsample_pack4 = 0;
    pipeline_gridsample_pack8 = 0;
    pipeline_gridsample_compute_offset = 0;
    permute_g = 0;
}

int GridSample_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // prepare for grid tensor permute layer
    if (shape.dims != 0)
    {
        permute_g = create_layer(LayerType::Permute);
        ParamDict pd;
        if (shape.dims == 3)
            pd.set(0, 4);
        else if (shape.dims == 4)
            pd.set(0, 18);
        permute_g->load_param(pd);
        permute_g->create_pipeline(opt);
    }

    int elempack = 1;
    if (shape.dims == 1 || shape.dims == 2)
        return -100;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;
    int out_elempack = 1;
    if (out_shape.dims == 1 || out_shape.dims == 2)
        return -100;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

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

    // create coord computing pipeline
    {
        const Mat& grid_shape = bottom_shapes.size() >= 2 ? bottom_shapes[1] : Mat();
        Mat local_size_xyz;
        if (grid_shape.dims == 3)
        {
            local_size_xyz.w = std::min(4, grid_shape.w);
            local_size_xyz.h = std::min(4, grid_shape.h);
            local_size_xyz.c = std::min(4, grid_shape.c);
        }
        if (grid_shape.dims == 4)
        {
            local_size_xyz.w = std::min(4, grid_shape.w);
            local_size_xyz.h = std::min(4, grid_shape.h * grid_shape.d);
            local_size_xyz.c = std::min(4, grid_shape.c);
        }

        LayerShaderType::LayerShaderType Shader_compute_offset;

        if (sample_type == 1)
        {
            Shader_compute_offset = bottom_shapes.empty() ? LayerShaderType::gridsample_bilinear_d3_compute_offset : bottom_shapes[0].dims == 3 ? LayerShaderType::gridsample_bilinear_d3_compute_offset
                                                                                           : LayerShaderType::gridsample_bilinear_d4_compute_offset;
        }
        else
        {
            Shader_compute_offset = sample_type == 2 ? LayerShaderType::gridsample_nearest_compute_offset
                                                     : LayerShaderType::gridsample_bicubic_compute_offset;
        }
        std::vector<vk_specialization_type> specializations(2 + 12);
        specializations[0].i = padding_mode;
        specializations[1].i = align_corner;
        specializations[2 + 0].i = shape.dims;
        specializations[2 + 1].i = shape.w;
        specializations[2 + 2].i = shape.h;
        specializations[2 + 3].i = shape.d;
        specializations[2 + 4].i = shape.c;
        specializations[2 + 5].i = shape.cstep;
        specializations[2 + 0].i = grid_shape.dims;
        specializations[2 + 1].i = grid_shape.w;
        specializations[2 + 2].i = grid_shape.h;
        specializations[2 + 3].i = grid_shape.d;
        specializations[2 + 4].i = grid_shape.c;
        specializations[2 + 5].i = grid_shape.cstep;
        pipeline_gridsample_compute_offset = new Pipeline(vkdev);
        pipeline_gridsample_compute_offset->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_compute_offset->create(Shader_compute_offset, opt, specializations);
        if (bottom_shapes.empty() && sample_type == 2)
        {
            pipeline_gridsample_bilinear_d4 = new Pipeline(vkdev);
            pipeline_gridsample_bilinear_d4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_gridsample_bilinear_d4->create(LayerShaderType::gridsample_bilinear_d4_compute_offset, opt, specializations);
            pipeline_gridsample_bilinear_d3 = new Pipeline(vkdev);
            pipeline_gridsample_bilinear_d3->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_gridsample_bilinear_d3->create(LayerShaderType::gridsample_bilinear_d3_compute_offset, opt, specializations);
        }
    }
    std::vector<vk_specialization_type> specializations(3 + 12);
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
        LayerShaderType::LayerShaderType gridsample = sample_type == 1 ? LayerShaderType::gridsample_nearest : sample_type == 2 ? LayerShaderType::gridsample_bilinear
                                                                                                                                : LayerShaderType::gridsample_bicubic;
        pipeline_gridsample = new Pipeline(vkdev);
        pipeline_gridsample->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample->create(gridsample, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        LayerShaderType::LayerShaderType gridsample_pack4 = sample_type == 1 ? LayerShaderType::gridsample_nearest_pack4 : sample_type == 2 ? LayerShaderType::gridsample_bilinear_pack4
                                                                                                                                      : LayerShaderType::gridsample_bicubic_pack4;
        pipeline_gridsample_pack4 = new Pipeline(vkdev);
        pipeline_gridsample_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_pack4->create(gridsample_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        LayerShaderType::LayerShaderType gridsample_pack8 = sample_type == 1 ? LayerShaderType::gridsample_nearest_pack8 : sample_type == 2 ? LayerShaderType::gridsample_bilinear_pack8
                                                                                                                                      : LayerShaderType::gridsample_bicubic_pack8;
        pipeline_gridsample_pack8 = new Pipeline(vkdev);
        pipeline_gridsample_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_pack8->create(gridsample_pack8, opt, specializations);
    }

    return 0;
}

int GridSample_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_gridsample;
    pipeline_gridsample = 0;

    delete pipeline_gridsample_pack4;
    pipeline_gridsample_pack4 = 0;

    delete pipeline_gridsample_pack8;
    pipeline_gridsample_pack8 = 0;

    delete pipeline_gridsample_compute_offset;
    pipeline_gridsample_compute_offset = 0;

    permute_g->destroy_pipeline(opt);
    delete permute_g;

    return 0;
}

int GridSample_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_blobs[0].elempack;

    ncnn::VkMat grid;

    // chw2whc(whc => hcw) or cwhd2whdc(whdc => hdcw)
    if (permute_g != 0)
    {
        permute_g->forward(bottom_blobs[1], grid, cmd, opt);
    }
    else
    {
        Layer* permute_rt = create_layer(LayerType::Permute);
        ParamDict pd;
        if (bottom_blobs[0].dims == 3)
            pd.set(0, 4);
        else if (bottom_blobs[0].dims == 4)
            pd.set(0, 18);
        permute_rt->load_param(pd);
        permute_rt->vkdev = this->vkdev;
        permute_rt->create_pipeline(opt);
        permute_rt->forward(bottom_blobs[1], grid, cmd, opt);
    }

    VkMat tmp_compute_blob;
    if (sample_type == 1)
    {
        tmp_compute_blob.create(grid.w, grid.h, 1, grid.elemsize, bottom_blobs[0].dims == 3 ? 4 : 8, opt.blob_vkallocator);
    }

    //get coord
    {
        std::vector<VkMat> bindings;
        if (sample_type == 1)
        {
            bindings.resize(2);
            bindings[0] = grid;
            bindings[1] = tmp_compute_blob;
        }
        else
        {
            bindings.resize(1);
            bindings[0] = grid;
        }

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blobs[0].dims;
        constants[1].i = bottom_blobs[0].w;
        constants[2].i = bottom_blobs[0].h;
        constants[3].i = bottom_blobs[0].d;
        constants[4].i = bottom_blobs[0].c;
        constants[5].i = bottom_blobs[0].cstep;
        constants[6].i = grid.dims;
        constants[7].i = grid.w;
        constants[8].i = grid.h;
        constants[9].i = grid.d;
        constants[10].i = grid.c;
        constants[11].i = grid.cstep;

        const Pipeline* pipeline = pipeline_gridsample_compute_offset;

        if (sample_type == 1)
        {
            const Pipeline* pipeline = bottom_blobs[0].dims == 3 ? pipeline_gridsample_bilinear_d3 : pipeline_gridsample_bilinear_d4;
            cmd.record_pipeline(pipeline, bindings, constants, tmp_compute_blob);
        }
        else
        {
            cmd.record_pipeline(pipeline, bindings, constants, grid);
        }
    }

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blobs[0];
    bindings[1] = grid;
    bindings[2] = bottom_blobs[1];

    std::vector<vk_constant_type> constants(12);
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

    ncnn::VkImageMat grid;

    // convert pack to 1
    if (bottom_blobs[1].elempack != 1)
    {
        permute_g->forward(bottom_blobs[1], grid, cmd, opt);
    }
    else
    {
        grid.create_like(bottom_blobs[1], opt.blob_vkallocator);
    }

    //get coord
    {
        std::vector<VkImageMat> bindings(1);
        bindings[0] = grid;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = grid.dims;
        constants[1].i = grid.w;
        constants[2].i = grid.h * grid.d;
        constants[3].i = grid.c;
        constants[4].i = 0; //grid_blob cstep

        const Pipeline* pipeline = pipeline_gridsample_compute_offset;

        cmd.record_pipeline(pipeline, bindings, constants, grid);
    }

    std::vector<VkImageMat> bindings(3);
    bindings[0] = bottom_blobs[0];
    bindings[1] = grid;
    bindings[2] = bottom_blobs[1];

    std::vector<vk_constant_type> constants(12);
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
