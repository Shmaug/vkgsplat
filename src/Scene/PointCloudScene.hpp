#pragma once

#include <json.hpp>
#include <Rose/RadixSort/RadixSort.hpp>

namespace vkgsplat {

using namespace RoseEngine;

struct PointCloudScene {
private:
	std::vector<ImageView> images;
	std::vector<float4x4>  viewTransformsCpu;
	std::vector<float4x4>  projectionTransformsCpu;
	BufferRange<float4x4>  viewTransforms;
	BufferRange<float4x4>  projectionTransforms;

	BufferRange<float3> vertices;
	BufferRange<float3> vertexColors;

public:
	// We store transmittance=1-alpha in the alpha channel
	static constexpr vk::PipelineColorBlendAttachmentState GetTransmittanceBlendState() {
		return vk::PipelineColorBlendAttachmentState {
			.blendEnable         = true,
			.srcColorBlendFactor = vk::BlendFactor::eDstAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOne,
			.colorBlendOp        = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eDstAlpha,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp        = vk::BlendOp::eAdd,
			.colorWriteMask      = vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags} };
	}

	inline const auto& GetImages() const { return images; }
	inline const auto& GetViewsCpu() const { return viewTransformsCpu; }
	inline const auto& GetProjectionsCpu() const { return projectionTransformsCpu; }
	inline const auto& GetViews() const { return viewTransforms; }
	inline const auto& GetProjections() const { return projectionTransforms; }

	inline const auto& GetVertices() const { return vertices; }
	inline const auto& GetVertexColors() const { return vertexColors; }

	inline ShaderParameter GetShaderParameter() const {
		ShaderParameter params = {};
		params["vertices"]    = (BufferParameter)vertices;
		params["colors"]      = (BufferParameter)vertexColors;
		params["views"]       = (BufferParameter)viewTransforms;
		params["projections"] = (BufferParameter)projectionTransforms;
		params["numVertices"] = (uint32_t)vertices.size();
		params["numViews"]    = (uint32_t)images.size();
		return params;
	}

	inline void Load(CommandContext& context, const std::filesystem::path& p) {
		using namespace nlohmann;

		auto json2float4x4 = [](const json& serialized) {
			float4x4 v;
			for (uint i = 0; i < 4; i++)
				for (uint j = 0; j < 4; j++)
					v[j][i] = serialized[i][j].get<float>();
			return v;
		};
		auto json2float3buf = [&](const json& serialized) {
			std::vector<float3>  data;
			data.reserve(serialized.size());
			for (const auto& v : serialized)
				data.emplace_back(
					v[0].get<float>(),
					v[1].get<float>(),
					v[2].get<float>());			
			return context.UploadData(data, vk::BufferUsageFlagBits::eStorageBuffer);
		};

		const std::filesystem::path imageDir = p.parent_path() / p.stem();

		images.clear();
		viewTransformsCpu.clear();
		projectionTransformsCpu.clear();
		
		std::ifstream fs(p);
		const json pointCloud = json::parse(fs);
		const json trainCameras = pointCloud["train_cameras"];
		const json testCameras  = pointCloud["test_cameras"];

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
							.queueFamilies = { context.QueueFamily() } }),
						vk::ImageSubresourceRange{
							.aspectMask = vk::ImageAspectFlagBits::eColor,
							.baseMipLevel = 0,
							.levelCount = 1,
							.baseArrayLayer = 0,
							.layerCount = 1 });
					if (img) context.Copy(d.data, img);
					images.emplace_back(img);
					viewTransformsCpu.emplace_back(json2float4x4(c["view"]));
					projectionTransformsCpu.emplace_back(json2float4x4(c["projection"]));
			}
		}

		viewTransforms       = context.UploadData(viewTransformsCpu,       vk::BufferUsageFlagBits::eStorageBuffer);
		projectionTransforms = context.UploadData(projectionTransformsCpu, vk::BufferUsageFlagBits::eStorageBuffer);

		vertices      = json2float3buf(pointCloud["points"]);
		vertexColors  = json2float3buf(pointCloud["colors"]);
	}
};

}