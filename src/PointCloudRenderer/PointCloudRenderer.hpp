#pragma once

#include <Rose/Core/CommandContext.hpp>
#include <Rose/Core/PipelineCache.hpp>
#include <Rose/Render/ViewportCamera.hpp>

#include "Scene/GaussianScene.hpp"

using namespace RoseEngine;

namespace vkgsplat {

struct PointCloudRenderer {
	PipelineCache createSortPairsPipeline = PipelineCache(FindShaderPath("CreateSortPairs.cs.slang"));
	PipelineCache rasterPointsPipeline = PipelineCache({
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "vsmain" },
		{ FindShaderPath("PointCloudRenderer.3d.slang"), "fsmain" }
	});

	float pointSize = 10.f;
    
	BufferRange<uint2> sortPairs;
	RadixSort radixSort;

    inline void DrawGui(CommandContext& context) {
        ImGui::DragFloat("Point size", &pointSize, .5f, 0.f, 4000.f);
    }

	inline void Render(
        CommandContext&       context,
        const ImageView&      renderTarget,
        const GaussianScene&  scene,
        const ViewportCamera& camera,
        const Transform&      sceneToWorld) {

        const uint2 renderExtent = renderTarget.Extent();

        const Transform worldToScene = inverse(sceneToWorld);
        const Transform cameraToWorld = camera.GetCameraToWorld();
        const Transform worldToCamera = inverse(cameraToWorld);
        const Transform sceneToCamera = worldToCamera * sceneToWorld;
        const Transform cameraToScene = inverse(sceneToCamera);
        const Transform projection = camera.GetProjection((float)renderExtent.x / (float)renderExtent.y);

		const float3 cameraPosScene = worldToScene.TransformPoint(camera.position);

		if (scene.GetVertices()) {
            const auto& vertices = scene.GetVertices();
            
            if (!sortPairs || sortPairs.size() != vertices.size())
                sortPairs = Buffer::Create(context.GetDevice(), vertices.size()*sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer);

            context.PushDebugLabel("Sort points");

            ShaderParameter params = {};
            params["sortPairs"] = (BufferParameter)sortPairs;
            params["vertices"]  = (BufferParameter)vertices;
            params["cameraPosition"] = cameraPosScene;
            params["cameraForward"]  = normalize(transpose(cameraToScene).TransformVector(float3(0,0,1)));
            params["vertexCount"] = (uint32_t)vertices.size();
            context.Dispatch(*createSortPairsPipeline.get(context.GetDevice()), vertices.size(), params);

            radixSort(context, sortPairs);
            
            context.PopDebugLabel();
        }

        context.AddBarrier(renderTarget, Image::ResourceState{
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
            .stage  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .access =  vk::AccessFlagBits2::eColorAttachmentRead|vk::AccessFlagBits2::eColorAttachmentWrite,
            .queueFamily = context.QueueFamily() });
        context.ExecuteBarriers();

		ShaderDefines defines;
        
        GraphicsPipelineInfo pipelineInfo {
            .vertexInputState = {},
            .inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{
                .topology = vk::PrimitiveTopology::ePointList },
            .rasterizationState = vk::PipelineRasterizationStateCreateInfo{
                .depthClampEnable = false,
                .rasterizerDiscardEnable = false,
                .polygonMode = vk::PolygonMode::ePoint,
                .cullMode = vk::CullModeFlagBits::eFront,
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
                .attachments = { GaussianScene::GetTransmittanceBlendState() } },
            .dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor },
            .dynamicRenderingState = DynamicRenderingState{
                .colorFormats = { renderTarget.GetImage()->Info().format } } };
        Pipeline& drawPipeline = *rasterPointsPipeline.get(context.GetDevice(), defines, pipelineInfo).get();
        auto drawDescriptorSets = context.GetDescriptorSets(*drawPipeline.Layout());

        // prepare draw parameters
        {

            ShaderParameter params = {};
            params["viewProjection"] = projection * sceneToCamera;
            params["cameraPosition"] = cameraPosScene;
			params["pointCloud"] = scene.GetPointCloudParams();
			params["sortPairs"] = (BufferParameter)sortPairs;
            params["pointSize"] = pointSize;

            context.UpdateDescriptorSets(*drawDescriptorSets, params, *drawPipeline.Layout());
        }

        // rasterization

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
		
		if (scene.GetVertices())
		{
            context->bindPipeline(vk::PipelineBindPoint::eGraphics, **drawPipeline);
            context.BindDescriptors(*drawPipeline.Layout(), *drawDescriptorSets);
            context->draw((uint32_t)scene.GetVertices().size(), 1, 0, 0);
		}
	
		context->endRendering();
	}
};

}