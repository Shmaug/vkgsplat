#pragma once

#include <Rose/Core/CommandContext.hpp>

namespace vkgsplat {

using namespace RoseEngine;

struct AdamOptimizer {
	ref<Pipeline> pipeline;

	BufferRange<float2> moments;

	float stepSize = 0.001f; // α
	float decay1 = 0.9f;     // β1
	float decay2 = 0.999f;   // β2
	uint32_t t = 0;

	inline void reset() { t = 0; }

	inline void operator()(CommandContext& context, const BufferRange<float>& parameters, const BufferRange<float>& gradients) {
		if (!pipeline) {
			ShaderDefines defs {
				{ "PARAMETER_TYPE",  },
			};
			auto shaderFile = FindShaderPath("Adam.cs.slang");
			pipeline = Pipeline::CreateCompute(context.GetDevice(), ShaderModule::Create(context.GetDevice(), shaderFile, "main", "sm_6_7", defs));
		}

		if (!moments || moments.size() != parameters.size()) {
			moments = Buffer::Create(context.GetDevice(), 2*parameters.size_bytes(), vk::BufferUsageFlagBits::eStorageBuffer);
			t = 0;
		}

		ShaderParameter params = {};
		params["parameterCount"] = (uint32_t)parameters.size();
		params["t"]  = t;
		params["decayRates"] = float2(decay1, decay2);
		params["stepSize"] = stepSize;
		params["parameters"] = (BufferView)parameters;
		params["gradients"]  = (BufferView)gradients;
		params["moments"]    = (BufferView)moments;

		context.Dispatch(*pipeline, (uint32_t)parameters.size());

		t++;
	}
};

}