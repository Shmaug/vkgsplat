#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>

#include "PointCloudRenderer/PointCloudRenderer.hpp"

using namespace vkgsplat;
using namespace RoseEngine;

int main(int argc, const char** argv) {
	WindowedApp app("GaussianRenderer", {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME
	});

	PipelineCache computeAlphaPipeline = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));

	PointCloudScene    scene;
	ViewportCamera     camera;
	PointCloudRenderer renderer;
		
	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;
	auto getSceneToWorld = [&]() { return Transform::Translate(sceneTranslation) * Transform::Rotate(glm::quat(sceneRotation)) * Transform::Scale(float3(sceneScale)); };

	int  selectedView = -1;
	bool renderSelectedView = true;

	ImageView viewportRenderTarget;
	ImageView inputViewRenderTarget;

	auto openSceneDialog = [&]() {
		auto f = pfd::open_file(
			"Choose scene",
			"",
			{ "JSON files (.json)", "*.json" },
			false
		);
		for (const std::string& filepath : f.result()) {
			scene.Load(app.CurrentContext(), filepath);
		}
	};

	if (argc > 1) {
		app.contexts[0]->Begin();
		scene.Load(*app.contexts[0], argv[1]);
		app.contexts[0]->Submit();
	}

	app.AddMenuItem("File", [&]() {
		if (ImGui::MenuItem("Open scene")) {
			openSceneDialog();
		}
	});

	app.AddWidget("Properties", [&]() {
		if (ImGui::CollapsingHeader("Camera")) {
			camera.DrawInspectorGui();
		}

		if (ImGui::CollapsingHeader("Scene")) {
			ImGui::Text("%u views", scene.GetImages().size());
			ImGui::Text("%u vertices", scene.GetVertices().size());
			ImGui::DragFloat3("Translation", &sceneTranslation.x, 0.1f);
			ImGui::DragFloat3("Rotation", &sceneRotation.x, float(M_1_PI)*0.1f, -float(M_PI), float(M_PI));
			ImGui::DragFloat("Scale", &sceneScale, 0.01f, 0.f, 1000.f);
			ImGui::Separator();
			// scene.DrawGui(context);
		}

		if (ImGui::CollapsingHeader("Renderer")) {
			if (viewportRenderTarget) {
				ImGui::Text("%u x %u", viewportRenderTarget.Extent().x, viewportRenderTarget.Extent().y);
			}
			renderer.DrawGui(app.CurrentContext());
		}
	}, true);

	app.AddWidget("Viewport", [&]() {
		if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiMod_Ctrl)) {
			openSceneDialog();
		}
		
		camera.Update(app.dt);

		auto& context = app.CurrentContext();

		const float2 extentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
		const uint2 extent = uint2(extentf);
		if (extent.x == 0 || extent.y == 0) return;

		if (!viewportRenderTarget || viewportRenderTarget.Extent().x != extent.x || viewportRenderTarget.Extent().y != extent.y) {
			viewportRenderTarget = ImageView::Create(
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

		// Draw the viewportRenderTarget image to the window
		ImGui::Image(Gui::GetTextureID(viewportRenderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(extentf));

		// render the scene into viewportRenderTarget
		
		context.PushDebugLabel("GaussianRenderer::Render");

        const Transform view = inverse(camera.GetCameraToWorld()) * getSceneToWorld();
		const Transform proj = camera.GetProjection(extentf.x / extentf.y);
		renderer.Render(context, viewportRenderTarget, scene, view, proj);
		
		// compute alpha = 1 - T
		{
			const uint2 extent = (uint2)viewportRenderTarget.Extent();
			ShaderParameter params = {};
			params["image"] = ImageParameter{ .image = viewportRenderTarget, .imageLayout = vk::ImageLayout::eGeneral };
			params["dim"] = extent;
			context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), extent, params);
		}

		context.PopDebugLabel();
	}, true, WindowedApp::WidgetFlagBits::eNoBorders);

	app.AddWidget("Input Views", [&]() {
		if (!scene.GetImages().empty())
		{
			ImGui::SetNextItemWidth(75);
			ImGui::DragInt("View", &selectedView, 1.f, -1, scene.GetImages().size()-1);
			if (selectedView >= scene.GetImages().size()) selectedView = scene.GetImages().size()-1;
			if (selectedView >= 0) {
				ImGui::SameLine();
				ImGui::Checkbox("Render view", &renderSelectedView);

				ImageView img = scene.GetImages()[selectedView];
				
				if (renderSelectedView) {
					CommandContext& context = app.CurrentContext();

					if (!inputViewRenderTarget || inputViewRenderTarget.Extent() != img.Extent()) {
						inputViewRenderTarget = ImageView::Create(
							Image::Create(context.GetDevice(), ImageInfo{
								.format = vk::Format::eR8G8B8A8Unorm,
								.extent = img.Extent(),
								.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
								.queueFamilies = { context.QueueFamily() } }),
							vk::ImageSubresourceRange{
								.aspectMask = vk::ImageAspectFlagBits::eColor,
								.baseMipLevel = 0,
								.levelCount = 1,
								.baseArrayLayer = 0,
								.layerCount = 1 });
					}

					context.PushDebugLabel("GaussianRenderer::Render");

					const Transform view = Transform{ scene.GetViewsCpu()[selectedView] };
					const Transform proj = Transform{ scene.GetProjectionsCpu()[selectedView] };
					renderer.Render(context, inputViewRenderTarget, scene, view, proj);
					
					// compute alpha = 1 - T
					{
						ShaderParameter params = {};
						params["image"] = ImageParameter{ .image = inputViewRenderTarget, .imageLayout = vk::ImageLayout::eGeneral };
						params["dim"] = uint2(img.Extent());
						context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), img.Extent(), params);
					}

					img = inputViewRenderTarget;
					context.PopDebugLabel();
				}

				// draw image to fill window width
				uint2 extent = uint2(std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin()));
				if (extent.x == 0 || extent.y == 0) return;

				// match img aspect
				extent.y = extent.x * (img.Extent().y / (float)img.Extent().x);
				
				ImGui::Image(Gui::GetTextureID(img), ImVec2(extent.x, extent.y));
			}
		}
	}, true);

	app.Run();

	app.device->Wait();

	return EXIT_SUCCESS;
}
