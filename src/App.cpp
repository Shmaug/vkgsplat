#include <Rose/Core/WindowedApp.hpp>
#include "Adam/Adam.hpp"
#include <portable-file-dialogs.h>

#include "PointCloudRenderer/PointCloudRenderer.hpp"

using namespace vkgsplat;
using namespace RoseEngine;

int main(int argc, const char** argv) {
	WindowedApp app("GaussianRenderer", {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
		VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
	});

	PipelineCache computeAlphaPipeline = PipelineCache(FindShaderPath("InvertAlpha.cs.slang"));

	PointCloudScene    scene;
	ViewportCamera     camera;
	PointCloudRenderer renderer;
	AdamOptimizer      adam;

	float3 sceneTranslation = float3(0);
	float3 sceneRotation = float3(0);
	float  sceneScale = 1.f;
	auto getSceneToWorld = [&]() { return Transform::Translate(sceneTranslation) * Transform::Rotate(glm::quat(sceneRotation)) * Transform::Scale(float3(sceneScale)); };

	int  selectedView = -1;
	bool showReference = true;

	float resolutionScale = 0.25f;
	float currentLoss = std::numeric_limits<float>::infinity();
	bool runOptimizer = false;
	// cache initial data so we can quickly restart optimization
	BufferRange<float3> initialVertices;
	BufferRange<float4> initialVertexColors;
	ImageView viewportRenderTarget;
	ImageView inputViewRenderTarget;
	ImageView optimizerRenderTarget;
	ImageView optimizerRefImg;
	std::queue<std::pair<BufferRange<float>, uint64_t>> lossCpuQueue;

	auto stepOptimizer = [&](CommandContext& context) {
		if (adam.t == 0)
			currentLoss = -1;

		BufferRange<float> lossBuf;
		BufferRange<float> lossCpu;
		if (!lossCpuQueue.empty()) {
			auto [lossCpu_, value] = lossCpuQueue.front();
			if (context.GetDevice().CurrentTimelineValue() >= value) {
				lossCpu = lossCpu_;
				currentLoss = (currentLoss < 0) ? lossCpu[0] : lerp(lossCpu[0], currentLoss, 0.9);
				lossCpuQueue.pop();
			}
		}
		if (!lossCpu) lossCpu = Buffer::Create(context.GetDevice(), sizeof(float), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent, VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
		lossCpuQueue.push({ lossCpu, context.GetDevice().NextTimelineSignal() });
		lossBuf = context.GetTransientBuffer<float>(1, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst);

		const uint32_t imageIndex = rand() % scene.numTrainCameras;
		const Transform view = Transform{ scene.viewTransformsCpu[imageIndex] };
		const Transform proj = Transform{ scene.projectionTransformsCpu[imageIndex] };

		ImageView refImg = scene.images[imageIndex];
		const uint2 scaledExtent = max(uint2(float2(refImg.Extent())*resolutionScale), uint2(1));
		const bool isExtentScaled = scaledExtent.x != refImg.Extent().x || scaledExtent.y != refImg.Extent().y;

		if (!optimizerRenderTarget || optimizerRenderTarget.Extent().x != scaledExtent.x || optimizerRenderTarget.Extent().y != scaledExtent.y) {
			optimizerRenderTarget = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR16G16B16A16Sfloat,
					.extent = uint3(scaledExtent, 1u),
					.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
					.queueFamilies = { context.QueueFamily() } }));
		}
		if (isExtentScaled) {
			if (!optimizerRefImg || optimizerRefImg.Extent().x != scaledExtent.x || optimizerRefImg.Extent().y != scaledExtent.y) {
				optimizerRefImg = ImageView::Create(
					Image::Create(context.GetDevice(), ImageInfo{
						.format = refImg.GetImage()->Info().format,
						.extent = uint3(scaledExtent, 1u),
						.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
						.queueFamilies = { context.QueueFamily() } }));
			}
		} else {
			optimizerRefImg = {};
		}
		
		if (isExtentScaled) {
			context.Blit(refImg, optimizerRefImg);
			refImg = optimizerRefImg;
		}

		context.Fill(lossBuf, 0.f);
		scene.pointCloud.vertices.clearGradients(context);
		scene.pointCloud.vertexColors.clearGradients(context);

		renderer.RenderGradients(context, optimizerRenderTarget, scene.pointCloud, view, proj, refImg, lossBuf);

		context.Copy(lossBuf, lossCpu);

		adam(context, scene.pointCloud.vertices);
		adam(context, scene.pointCloud.vertexColors);
		adam.increment();
	};

	auto openSceneDialog = [&]() {
		auto& context = app.CurrentContext();
		auto f = pfd::open_file(
			"Choose scene",
			"",
			{ "JSON files (.json)", "*.json" },
			false
		);
		for (const std::string& filepath : f.result()) {
			app.device->Wait();

			scene.Load(context, filepath);

			// backup initial data
			initialVertices     = Buffer::Create(context.GetDevice(), scene.pointCloud.vertices.data.size_bytes(),     vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc);
			initialVertexColors = Buffer::Create(context.GetDevice(), scene.pointCloud.vertexColors.data.size_bytes(), vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc);
			context.Copy(scene.pointCloud.vertices.data,     initialVertices);
			context.Copy(scene.pointCloud.vertexColors.data, initialVertexColors);

			adam.reset();
		}
	};

	// load input scene
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
			ImGui::Text("%u views (%u train)", scene.images.size(), scene.numTrainCameras);
			ImGui::Text("%u vertices", scene.pointCloud.size());
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

		if (ImGui::CollapsingHeader("Optimizer")) {
			ImGui::Checkbox("Run", &runOptimizer);
			ImGui::SameLine();
			if (ImGui::Button("Reset")) {
				adam.reset();
				if (scene.pointCloud.vertices) {
					// restore initial data
					app.CurrentContext().Copy(initialVertices,     scene.pointCloud.vertices.data);
					app.CurrentContext().Copy(initialVertexColors, scene.pointCloud.vertexColors.data);
				}
			}
			
			ImGui::DragFloat("Step size", &adam.stepSize, 0.001f, 0, 1.f);
			ImGui::DragFloat2("Decay rates", &adam.decay1, 0.001f, 0, 1.0f - 1e-6f);

			if (ImGui::SliderFloat("Resolution scale", &resolutionScale, 0.f, 1.f)) app.device->Wait();

			const uint2 extent = scene.images.empty() ? uint2(0) : uint2(scene.images[0].Extent());
			const uint2 scaledExtent = max(uint2(float2(extent)*resolutionScale), uint2(1));
			const auto&[number,unit] = FormatNumber(scaledExtent.x * scaledExtent.y);
			ImGui::Text("%u x %u (%.2f%s pixels)", scaledExtent.x, scaledExtent.y, number, unit);

			ImGui::Text("Iteration: %u, loss: %f", adam.t, currentLoss);
		}
	}, true);

	app.AddWidget("Viewport", [&]() {
		if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiMod_Ctrl)) {
			openSceneDialog();
		}
		
		camera.Update(app.dt);

		auto& context = app.CurrentContext();

		if (runOptimizer && !scene.images.empty()) stepOptimizer(context);

		const float2 extentf = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
		const uint2 extent = uint2(extentf);
		if (extent.x == 0 || extent.y == 0) return;

		if (!viewportRenderTarget || viewportRenderTarget.Extent().x != extent.x || viewportRenderTarget.Extent().y != extent.y) {
			viewportRenderTarget = ImageView::Create(
				Image::Create(context.GetDevice(), ImageInfo{
					.format = vk::Format::eR16G16B16A16Sfloat,
					.extent = uint3(extent, 1),
					.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
					.queueFamilies = { context.QueueFamily() } }));
		}

		// Draw the viewportRenderTarget image to the window
		ImGui::Image(Gui::GetTextureID(viewportRenderTarget, vk::Filter::eNearest), std::bit_cast<ImVec2>(extentf));

		// render the scene into viewportRenderTarget
		
		context.PushDebugLabel("App::Render");

        const Transform view = inverse(camera.GetCameraToWorld()) * getSceneToWorld();
		const Transform proj = camera.GetProjection(extentf.x / extentf.y);
		renderer.Render(context, viewportRenderTarget, scene.pointCloud, view, proj);
		
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
		auto& context = app.CurrentContext();
		if (!scene.images.empty()) {
			ImGui::SetNextItemWidth(75);
			ImGui::DragInt("View", &selectedView, 1.f, -1, scene.images.size()-1);
			if (selectedView >= scene.images.size()) selectedView = scene.images.size()-1;
			if (selectedView >= 0) {
				ImageView img = scene.images[selectedView];

				ImGui::SameLine();
				ImGui::Checkbox("Show reference", &showReference);

				if (!showReference) {
					if (!inputViewRenderTarget || inputViewRenderTarget.Extent().x != img.Extent().x || inputViewRenderTarget.Extent().y != img.Extent().y) {
						inputViewRenderTarget = ImageView::Create(
							Image::Create(context.GetDevice(), ImageInfo{
								.format = vk::Format::eR16G16B16A16Sfloat,
								.extent = img.Extent(),
								.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
								.queueFamilies = { context.QueueFamily() } }));
					}

					context.PushDebugLabel("InputViewWidget::Render");

					const Transform view = Transform{ scene.viewTransformsCpu[selectedView] };
					const Transform proj = Transform{ scene.projectionTransformsCpu[selectedView] };
					renderer.Render(context, inputViewRenderTarget, scene.pointCloud, view, proj);

					// compute alpha = 1 - T
					{
						ShaderParameter params = {};
						params["image"] = ImageParameter{ .image = inputViewRenderTarget, .imageLayout = vk::ImageLayout::eGeneral };
						params["dim"] = uint2(inputViewRenderTarget.Extent());
						context.Dispatch(*computeAlphaPipeline.get(context.GetDevice()), inputViewRenderTarget.Extent(), params);
					}

					context.PopDebugLabel();

					img = inputViewRenderTarget;
				}

				// draw image to fill window width
				uint2 extent = uint2(std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin()));
				if (extent.x == 0 || extent.y == 0) return;

				// match img aspect
				extent.y = extent.x * (img.Extent().y / (float)img.Extent().x);
				
				ImGui::Image(Gui::GetTextureID(img), ImVec2(extent.x, extent.y));
			}
		}
	}, true, WindowedApp::WidgetFlagBits::eNoBorders);

	app.Run();

	app.device->Wait();

	return EXIT_SUCCESS;
}
