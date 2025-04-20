#pragma once

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/Render/ViewportCamera.hpp>

#include "Scene/PointCloudScene.hpp"

using namespace RoseEngine;

namespace vkgsplat {

struct PointCloudRenderer {
	PipelineCache createSortPairs = PipelineCache(FindShaderPath("CreateSortPairs.cs.slang"));
	PipelineCache rasterPoints = PipelineCache({
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "meshmain" },
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "fsmain" }
	});
	PipelineCache computeRender    = PipelineCache(FindShaderPath("PointCloudRenderer.cs.slang"), "render");
	PipelineCache computeRenderBwd = PipelineCache(FindShaderPath("PointCloudRenderer.cs.slang"), "__bwd_render");
	float pointSize = 0.05f;
	float percentToDraw = 1.0f;
    
	RadixSort radixSort;

    inline void DrawGui(CommandContext& context) {
        ImGui::DragFloat("Point size", &pointSize, .01f, 0.f, 4000.f);
        ImGui::SliderFloat("Amount to draw", &percentToDraw, 0.f, 1.f);
    }

    inline BufferRange<uint2> Sort(CommandContext& context, const PointCloud& pointCloud, const Transform& sceneToCamera, const Transform& projection) {
        const uint32_t vertexCount = (uint32_t)pointCloud.size();
        BufferRange<uint2> sortPairs = context.GetTransientBuffer<uint2>(vertexCount, vk::BufferUsageFlagBits::eStorageBuffer);
    
        context.PushDebugLabel("Sort points");
        ShaderParameter params = {};
        params["sortPairs"] = (BufferParameter)sortPairs;
        params["vertices"]  = (BufferParameter)pointCloud.vertices.data;
        params["view"] = sceneToCamera.transform;
        params["vertexCount"] = vertexCount;
        params["zSign"] = (int32_t)(projection.transform[2][2] > 0 ? 1 : -1);
        createSortPairs(context, uint3(vertexCount,1,1), params);
        radixSort(context, sortPairs);
        context.PopDebugLabel();

        return sortPairs;
    }
    
	inline void Render(
        CommandContext&   context,
        const ImageView&  renderTarget,
        const PointCloud& pointCloud,
        const Transform&  sceneToCamera,
        const Transform&  projection,
        BufferRange<uint2> sortPairs = {}) {
        const uint32_t vertexCount = (uint32_t)pointCloud.size();
		if (vertexCount == 0)
        {
            context.ClearColor(renderTarget, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 1 }});
            return;
        }

        if (!sortPairs) sortPairs = Sort(context, pointCloud, sceneToCamera, projection);
        
        const uint2 renderExtent = renderTarget.Extent();

        // prepare draw pipeline

        ShaderDefines defines;
        
        GraphicsPipelineInfo pipelineInfo {
            .vertexInputState = {},
            .inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
                .topology = vk::PrimitiveTopology::eTriangleList },
            .rasterizationState = vk::PipelineRasterizationStateCreateInfo{
                .depthClampEnable = false,
                .rasterizerDiscardEnable = false,
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .depthBiasEnable = false },
            .multisampleState = vk::PipelineMultisampleStateCreateInfo{},
            .depthStencilState = vk::PipelineDepthStencilStateCreateInfo{
                .depthTestEnable = false,
                .depthWriteEnable = false,
                .depthCompareOp = vk::CompareOp::eLess,
                .depthBoundsTestEnable = false,
                .stencilTestEnable = false },
            .viewports = { vk::Viewport{} },
            .scissors = { vk::Rect2D{} },
            .colorBlendState = ColorBlendState{
                .attachments = { vk::PipelineColorBlendAttachmentState {
                    .blendEnable         = true,
                    .srcColorBlendFactor = vk::BlendFactor::eDstAlpha,
                    .dstColorBlendFactor = vk::BlendFactor::eOne,
                    .colorBlendOp        = vk::BlendOp::eAdd,
                    .srcAlphaBlendFactor = vk::BlendFactor::eDstAlpha,
                    .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                    .alphaBlendOp        = vk::BlendOp::eAdd,
                    .colorWriteMask      = vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags} } } },
            .dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
            .dynamicRenderingState = DynamicRenderingState{
                .colorFormats = { renderTarget.GetImage()->Info().format } } };
        Pipeline& drawPipeline = *rasterPoints.get(context.GetDevice(), defines, pipelineInfo).get();
        auto drawDescriptorSets = context.GetDescriptorSets(*drawPipeline.Layout());

        // prepare draw parameters
        {
            Transform t = transpose(sceneToCamera);
            ShaderParameter params = {};
            params["view"] = sceneToCamera;
            params["projection"] = projection;
            params["cameraRight"] = t.TransformVector(float3(1,0,0));
            params["cameraUp"] = t.TransformVector(float3(0,1,0));
            params["outputExtent"] = renderExtent;
            params["pointCloud"] = pointCloud.GetShaderParameter();
            params["pointCloud"]["numVertices"] = (uint32_t)(percentToDraw*vertexCount);
            params["sortPairs"] = (BufferParameter)sortPairs;
            params["pointSize"] = pointSize;

            context.UpdateDescriptorSets(*drawDescriptorSets, params, *drawPipeline.Layout());
        }

        // rasterize points

        context.AddBarrier(renderTarget, Image::ResourceState{
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
            .stage  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .access =  vk::AccessFlagBits2::eColorAttachmentRead|vk::AccessFlagBits2::eColorAttachmentWrite,
            .queueFamily = context.QueueFamily() });
        context.ExecuteBarriers();

        vk::RenderingAttachmentInfo attachments[1] = {
            vk::RenderingAttachmentInfo {
                .imageView = *renderTarget,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .resolveMode = vk::ResolveModeFlagBits::eNone,
                .resolveImageView = {},
                .resolveImageLayout = vk::ImageLayout::eUndefined,
                .loadOp  = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 1 }} } } };
        context->beginRendering(vk::RenderingInfo {
            .renderArea = vk::Rect2D{ {0, 0},  { renderExtent.x, renderExtent.y } },
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = attachments }); 
        context->setViewport(0, vk::Viewport{ 0, 0, (float)renderExtent.x, (float)renderExtent.y, 0, 1});
        context->setScissor(0,  vk::Rect2D{ {0, 0}, { renderExtent.x, renderExtent.y }});
    
        context->bindPipeline(vk::PipelineBindPoint::eGraphics, **drawPipeline);
        context.BindDescriptors(*drawPipeline.Layout(), *drawDescriptorSets);

        const uint32_t vertsPerGroup = drawPipeline.GetShader()->WorkgroupSize().x/4;
        context->drawMeshTasksEXT((uint32_t(vertexCount*percentToDraw) + vertsPerGroup-1) / vertsPerGroup, 1, 1);
    
        context->endRendering();
	}

	inline void RenderGradients(
        CommandContext&   context,
        const ImageView&  renderTarget,
        const PointCloud& pointCloud,
        const Transform&  sceneToCamera,
        const Transform&  projection,
        const ImageView&  referenceImage,
        const BufferRange<float>& loss) {
        const uint32_t vertexCount = (uint32_t)pointCloud.size();
		if (vertexCount == 0)
        {
            context.ClearColor(renderTarget, vk::ClearColorValue{std::array<float,4>{ 0, 0, 0, 1 }});
            return;
        }
        
        // sort points
        BufferRange<uint2> sortPairs = Sort(context, pointCloud, sceneToCamera, projection);
    
        const uint2 renderExtent = renderTarget.Extent();

        ImageView pixelVertexCounts = ImageView::Create(context.GetTransientImage(
            uint3(renderExtent, 1),
            vk::Format::eR32Uint,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage));

        ShaderParameter params = {};
        params["pointCloud"] = pointCloud.GetShaderParameter();
        params["pointCloud"]["numVertices"] = (uint32_t)(percentToDraw*vertexCount);
        params["sortPairs"] = (BufferParameter)sortPairs;
        params["outputColor"] = ImageParameter{.image = renderTarget,   .imageLayout = vk::ImageLayout::eGeneral};
        params["reference" ]  = ImageParameter{.image = referenceImage, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        params["outputLoss"] = (BufferParameter)loss;
        params["pixelVertexCounts"] = ImageParameter{.image = pixelVertexCounts, .imageLayout = vk::ImageLayout::eGeneral};
        params["view"] = sceneToCamera;
        params["projection"] = projection;
        params["outputExtent"] = renderExtent;
        params["pointSize"] = pointSize;

        // forward pass
        computeRender(context, uint3(renderExtent,1u), params);

        // backward pass
        computeRenderBwd(context, uint3(renderExtent,1u), params, { { "OUTPUT_LOSS", loss ? "1" : "0" } });
    }
};

}