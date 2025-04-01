#include <Rose/Core/WindowedApp.hpp>
#include <portable-file-dialogs.h>

#include "GaussianRenderer.hpp"

using namespace vkgsplat;

int main(int argc, const char** argv) {
	WindowedApp app("GaussianRenderer", {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME
	});

	GaussianRenderer renderer;

	auto openSceneDialog = [&]() {
		auto f = pfd::open_file(
			"Choose scene",
			"",
			{ "JSON files (.json)", "*.json" },
			false
		);
		for (const std::string& filepath : f.result()) {
			renderer.LoadScene(app.CurrentContext(), filepath);
		}
	};

	if (argc > 1) {
		app.contexts[0]->Begin();
		renderer.LoadScene(*app.contexts[0], argv[1]);
		app.contexts[0]->Submit();
	}

	app.AddMenuItem("File", [&]() {
		if (ImGui::MenuItem("Open scene")) {
			openSceneDialog();
		}
	});

	app.AddWidget("Properties", [&]() {
		renderer.DrawPropertiesGui(*app.contexts[app.swapchain->ImageIndex()]);
	}, true);

	app.AddWidget("Viewport", [&]() {
		if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiMod_Ctrl)) {
			openSceneDialog();
		}
		renderer.DrawWidgetGui(*app.contexts[app.swapchain->ImageIndex()], app.dt);
	}, true, WindowedApp::WidgetFlagBits::eNoBorders);

	app.Run();

	app.device->Wait();

	return EXIT_SUCCESS;
}
