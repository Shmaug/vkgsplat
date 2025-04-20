#include "SortUtils.h"
import Rose.Core.MathUtils;

using namespace vkgsplat;

ByteAddressBuffer vertices;
uniform float4x4 view;
uniform int zSign;
uniform uint vertexCount;

RWStructuredBuffer<uint2> sortPairs;

float getSortKey(const uint vertexId) {
    float3 vertex = vertices.Load<float3>(vertexId * sizeof(float3));
    vertex = mul(view, float4(vertex, 1)).xyz;
    return zSign * vertex.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    const uint vertexId = threadId.x;
    if (vertexId >= vertexCount)
        return;

    const float keyf = getSortKey(vertexId);
    const uint  key = (keyf != keyf || isnan(keyf) || isinf(keyf) || keyf == FLT_MAX) ? UINT32_MAX : order_preserving_float_map(keyf);

    sortPairs[vertexId] = uint2(key, vertexId);
}
