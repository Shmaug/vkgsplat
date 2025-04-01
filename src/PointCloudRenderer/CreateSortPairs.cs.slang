#include "SortUtils.h"
import Rose.Core.MathUtils;

using namespace vkgsplat;

ByteAddressBuffer vertices;
uniform float3 cameraPosition;
uniform float3 cameraForward;
uniform uint   vertexCount;

RWStructuredBuffer<uint2> sortPairs;

float getSortKey(const uint vertexId) {
    const float3 vertex = vertices.Load<float3>(vertexId * sizeof(float3));
    const float depth = dot(cameraForward, vertex - cameraPosition);
    return abs(depth);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    const uint vertexId = threadId.x;
    if (vertexId >= vertexCount)
        return;

    const float keyf = getSortKey(vertexId);
    const uint  key = (keyf != keyf || isnan(keyf) || isinf(keyf)) ? UINT32_MAX : order_preserving_float_map(keyf);

    sortPairs[vertexId] = uint2(key, vertexId);
}
