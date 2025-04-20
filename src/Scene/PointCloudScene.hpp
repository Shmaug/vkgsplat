#pragma once

#include <json.hpp>
#include <Rose/RadixSort/RadixSort.hpp>

namespace vkgsplat {

using namespace RoseEngine;

struct PointCloud {
	BufferRange<float3> vertices;
	BufferRange<float4> vertexColors;

	inline ShaderParameter GetShaderParameter() const {
		ShaderParameter params = {};
		params["vertices"]    = (BufferParameter)vertices;
		params["colors"]      = (BufferParameter)vertexColors;
		params["numVertices"] = (uint32_t)vertices.size();
		return params;
	}
};

struct PointCloudScene {
	std::vector<ImageView> images;
	std::vector<float4x4>  viewTransformsCpu;
	std::vector<float4x4>  projectionTransformsCpu;
	BufferRange<float4x4>  viewTransforms;
	BufferRange<float4x4>  projectionTransforms;

	PointCloud pointCloud;
	uint32_t numTrainCameras;

	inline void Load(CommandContext& context, const std::filesystem::path& p) {
		using namespace nlohmann;

		auto json2float4x4 = [](const json& serialized) {
			float4x4 v;
			for (uint i = 0; i < 4; i++)
				for (uint j = 0; j < 4; j++)
					v[j][i] = serialized[i][j].get<float>();
			return v;
		};

		const std::filesystem::path imageDir = p.parent_path() / p.stem();

		images.clear();
		viewTransformsCpu.clear();
		projectionTransformsCpu.clear();
		
		std::ifstream fs(p);
		const json pointCloudData = json::parse(fs);
		const json trainCameras = pointCloudData["train_cameras"];
		const json testCameras  = pointCloudData["test_cameras"];

		numTrainCameras = (uint32_t)trainCameras.size();

		images.reserve(trainCameras.size() + testCameras.size());
		viewTransformsCpu.reserve(trainCameras.size() + testCameras.size());
		projectionTransformsCpu.reserve(trainCameras.size() + testCameras.size());

		for (const auto& cameras : { trainCameras, testCameras }) {
			for (const auto& c : cameras) {				
					const PixelData d = LoadImageFile(context, imageDir / (c["image_name"].get<std::string>() + ".JPG"), false);
					if (!d.data) continue;
					const ImageView img = ImageView::Create(
						Image::Create(context.GetDevice(), ImageInfo{
							.format = d.format,
							.extent = d.extent,
							.mipLevels = 1,
							.queueFamilies = { context.QueueFamily() } }));
					if (img) context.Copy(d.data, img);
					images.emplace_back(img);
					viewTransformsCpu.emplace_back(json2float4x4(c["view"]));
					projectionTransformsCpu.emplace_back(json2float4x4(c["projection"]));
			}
		}

		viewTransforms       = context.UploadData(viewTransformsCpu,       vk::BufferUsageFlagBits::eStorageBuffer);
		projectionTransforms = context.UploadData(projectionTransformsCpu, vk::BufferUsageFlagBits::eStorageBuffer);

		std::vector<float3> vertices;
		std::vector<float4> vertexColors;
		vertices.reserve(pointCloudData["points"].size());
		vertexColors.reserve(pointCloudData["colors"].size());
		for (const auto& v : pointCloudData["points"]) vertices    .emplace_back(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
		for (const auto& v : pointCloudData["colors"]) vertexColors.emplace_back(v[0].get<float>(), v[1].get<float>(), v[2].get<float>(), 1.0f);
		
		pointCloud.vertices     = context.UploadData(vertices,     vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
		pointCloud.vertexColors = context.UploadData(vertexColors, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
	}
};

}