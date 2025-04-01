#pragma once

#include <stack>

#include "Scene/GaussianScene.hpp"
#include "PointCloudRenderer/PointCloudRenderer.hpp"

namespace vkgsplat {

using namespace RoseEngine;

class GaussianRenderer {
private:
	PipelineCache computeAlphaPipeline = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));

	ViewportCamera camera;	
	GaussianScene scene;
	PointCloudRenderer renderer;
	
	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;

	ImageView renderTarget;

public:
	inline void LoadScene(CommandContext& context, const std::filesystem::path& p) {
		scene.Load(context, p);
	}

	inline void DrawPropertiesGui(CommandContext& context) {
		if (ImGui::CollapsingHeader("Camera")) {
			camera.DrawInspectorGui();
		}

		if (ImGui::CollapsingHeader("Scene")) {
			ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
			ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
			ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
			// scene.DrawGui(context);
		}

		if (ImGui::CollapsingHeader("Renderer")) {
			if (renderTarget) {
				ImGui::Text("%u x %u", renderTarget.Extent().x, renderTarget.Extent().y);
			}
			renderer.DrawGui(context);
		}
	}

	inline void DrawWidgetGui(CommandContext& context, const double dt) {
		const float2 extentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
		const uint2 extent = uint2(extentf);
		if (extent.x == 0 || extent.y == 0) return;

		if (!renderTarget || renderTarget.Extent().x != extent.x || renderTarget.Extent().y != extent.y) {
			renderTarget = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR8G8B8A8Unorm,
					.extent = uint3(extent, 1),
					.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
					.queueFamilies = { context.QueueFamily() } }),
				vk::ImageSubresourceRange{
					.aspectMask = vk::ImageAspectFlagBits::eColor,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 });
		}

		// Draw the renderTarget image to the window
		ImGui::Image(Gui::GetTextureID(renderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(extentf));

		camera.Update(dt);

		// render the scene into renderTarget
		
		context.PushDebugLabel("GaussianRenderer::Render");

		const Transform sceneToWorld = Transform::Translate(sceneTranslation) * Transform::Rotate(glm::quat(sceneRotation)) * Transform::Scale(float3(sceneScale));
		renderer.Render(context, renderTarget, scene, camera, sceneToWorld);
		
		// compute alpha = 1 - T
		{
			const uint2 extent = (uint2)renderTarget.Extent();
			ShaderParameter params = {};
			params["image"] = ImageParameter{ .image = renderTarget, .imageLayout = vk::ImageLayout::eGeneral };
			params["dim"] = extent;
			context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), extent, params);
		}

		context.PopDebugLabel();
	}
};

}