#pragma once

#include <json.hpp>
#include <Rose/RadixSort/RadixSort.hpp>

namespace vkgsplat {

using namespace RoseEngine;

struct GaussianScene {
private:
	BufferRange<float3> vertices;
	BufferRange<float3> vertexColors;
	BufferRange<float3> vertexNormals;

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

	inline const BufferRange<float3>& GetVertices() const { return vertices; }
	inline const BufferRange<float3>& GetVertexColors() const { return vertexColors; }

	inline ShaderParameter GetPointCloudParams() const {
		ShaderParameter params = {};
		params["vertices"] = (BufferParameter)vertices;
		params["colors"]   = (BufferParameter)vertexColors;
		return params;
	}

	inline void Load(CommandContext& context, const std::filesystem::path& p) {
		using namespace nlohmann;

		auto json2float4x4 = [](const json& serialized) {
			float4x4 v;
			for (uint i = 0; i < 4; i++)
				for (uint j = 0; j < 4; j++)
					v[i][j] = serialized[i][j].get<float>();
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

		std::ifstream fs(p);
		json pointCloudJson = json::parse(fs);

		json trainCameras = pointCloudJson["train_cameras"];
		
		std::vector<ImageView> images;
		std::vector<float4x4>  views;
		std::vector<float4x4>  projections;
		images.reserve(trainCameras.size());
		views.reserve(trainCameras.size());
		projections.reserve(trainCameras.size());
		for (const auto& c : trainCameras) {				
				c["image_name"];

				views.emplace_back(json2float4x4(c["view"]));
				projections.emplace_back(json2float4x4(c["projection"]));
		}


		vertices      = json2float3buf(pointCloudJson["points"]);
		vertexColors  = json2float3buf(pointCloudJson["colors"]);
		vertexNormals = json2float3buf(pointCloudJson["normals"]);
	}
};

}