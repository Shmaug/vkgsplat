#pragma once

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/Render/ViewportCamera.hpp>

#include "Scene/PointCloudScene.hpp"

using namespace RoseEngine;

namespace vkgsplat {

struct PointCloudRenderer {
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("CreateSortPairs.cs.slang"));
	PipelineCache rasterPointsPipeline = PipelineCache({
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "meshmain" },
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "fsmain" }
	});

	float pointSize = 0.01f;
    
	RadixSort radixSort;

    inline void DrawGui(CommandContext& context) {
        ImGui::DragFloat("Point size", &pointSize, .01f, 0.f, 4000.f);
    }

	inline void Render(
        CommandContext&        context,
        const ImageView&       renderTarget,
        const PointCloudScene& scene,
        const Transform&       sceneToCamera,
        const Transform&       projection) {

        // sort points by distance to camera

        BufferRange<uint2> sortPairs;

        const uint32_t vertexCount = (uint32_t)scene.GetVertices().size();
		if (vertexCount > 0) {
	        sortPairs = context.GetTransientBuffer<uint2>(vertexCount, vk::BufferUsageFlagBits::eStorageBuffer);
            context.PushDebugLabel("Sort points");

            ShaderParameter params = {};
            params["sortPairs"] = (BufferParameter)sortPairs;
            params["vertices"]  = (BufferParameter)scene.GetVertices();
            params["cameraPosition"] = inverse(sceneToCamera).TransformPoint(float3(0));
            params["vertexCount"] = vertexCount;
            context.Dispatch(*createSortPairsPipeline.get(context.GetDevice()), vertexCount, params);

            radixSort(context, sortPairs);
            
            context.PopDebugLabel();
        }

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
                .attachments = { PointCloudScene::GetTransmittanceBlendState() } },
            .dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
            .dynamicRenderingState = DynamicRenderingState{
                .colorFormats = { renderTarget.GetImage()->Info().format } } };
        Pipeline& drawPipeline = *rasterPointsPipeline.get(context.GetDevice(), defines, pipelineInfo).get();
        auto drawDescriptorSets = context.GetDescriptorSets(*drawPipeline.Layout());

        // prepare draw parameters
        {
            Transform t = transpose(sceneToCamera);
            ShaderParameter params = {};
            params["viewProjection"] = projection * sceneToCamera;
            params["cameraUp"] = t.TransformVector(float3(0,1,0));
            params["cameraRight"] = t.TransformVector(float3(1,0,0));
			params["pointCloud"] = scene.GetShaderParameter();
			params["sortPairs"] = (BufferParameter)sortPairs;
            params["pointSize"] = pointSize;

            context.UpdateDescriptorSets(*drawDescriptorSets, params, *drawPipeline.Layout());
        }

        // rasterize points

        const uint2 renderExtent = renderTarget.Extent();

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
		
		if (vertexCount > 0)
		{
            context->bindPipeline(vk::PipelineBindPoint::eGraphics, **drawPipeline);
            context.BindDescriptors(*drawPipeline.Layout(), *drawDescriptorSets);

            const uint32_t vertsPerGroup = drawPipeline.GetShader()->WorkgroupSize().x/4;
            context->drawMeshTasksEXT((vertexCount + vertsPerGroup-1) / vertsPerGroup, 1, 1);
		}
	
		context->endRendering();
	}
};

}