#pragma once

#include <Rose/Core/CommandContext.hpp>
#include "BufferGradient.hpp"

namespace vkgsplat {

using namespace RoseEngine;

struct AdamOptimizer {
	std::vector<ref<Pipeline>> pipelines;

	float stepSize = 0.001f; // α
	float decay1 = 0.9f;     // β1
	float decay2 = 0.999f;   // β2
	uint32_t t = 0;

	inline void reset() { t = 0; }
	inline void increment() { t++; }

	template<int N>
	inline void operator()(CommandContext& context, const BufferGradient<N>& parameters) {
		if (pipelines.size() <= N) pipelines.resize(N+1);
		
		ref<Pipeline>& pipeline = pipelines[N];
		if (!pipeline) pipeline = Pipeline::CreateCompute(context.GetDevice(), ShaderModule::Create(context.GetDevice(), FindShaderPath("Adam.cs.slang"), "main", "sm_6_7", { { "NUM_CHANNELS", std::to_string(N) } }));
		
		ShaderParameter params = {};
		params["parameters"] = parameters.GetShaderParameter();
		params["parameterCount"] = (uint32_t)parameters.size();
		params["stepSize"] = stepSize;
		params["decayRates"] = float2(decay1, decay2);
		params["t"]  = t;
		context.Dispatch(*pipeline, (uint32_t)parameters.size(), params);
	}
};

}