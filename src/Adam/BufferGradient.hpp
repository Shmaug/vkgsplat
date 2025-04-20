#pragma once

#include <Rose/Core/CommandContext.hpp>

namespace vkgsplat {

using namespace RoseEngine;

template<int N>
struct BufferGradient {
    using T = glm::vec<N,float>;

    BufferRange<T> data;
    BufferRange<T> gradients;
    BufferRange<T> moments1;
    BufferRange<T> moments2;

    inline operator bool() const { return data; }
    inline vk::DeviceSize size() const { return data.size(); }

    inline void clearGradients(CommandContext& context) const { context.Fill(gradients.cast<float>(), 0.f); }

    inline ShaderParameter GetShaderParameter() const {
        ShaderParameter params = {};
        params["data"]      = (BufferParameter)data;
        params["gradients"] = (BufferParameter)gradients;
        params["moments1"]  = (BufferParameter)moments1;
        params["moments2"]  = (BufferParameter)moments2;
        return params;
    }
};

}