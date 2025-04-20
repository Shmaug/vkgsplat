#pragma once

#include <Rose/Core/CommandContext.hpp>

namespace vkgsplat {

using namespace RoseEngine;

struct AdamOptimizer {
	ref<Pipeline> pipeline;

	float stepSize = 0.001f; // α
	float decay1 = 0.9f;     // β1
	float decay2 = 0.999f;   // β2
	uint32_t t = 0;

	inline void reset() { t = 0; }

	inline void operator()(CommandContext& context, const BufferRange<float>& parameters, const BufferRange<float>& gradients, const BufferRange<float2>& moments) {
		if (!pipeline) pipeline = Pipeline::CreateCompute(context.GetDevice(), ShaderModule::Create(context.GetDevice(), FindShaderPath("Adam.cs.slang"), "main", "sm_6_7"));
		
		ShaderParameter params = {};
		params["parameterCount"] = (uint32_t)parameters.size();
		params["t"]  = t;
		params["decayRates"] = float2(decay1, decay2);
		params["stepSize"] = stepSize;
		params["parameters"] = (BufferView)parameters;
		params["gradients"]  = (BufferView)gradients;
		params["moments"]    = (BufferView)moments;
		context.Dispatch(*pipeline, (uint32_t)parameters.size(), params);

		t++;
	}
};

}