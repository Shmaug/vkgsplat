import Scene.PointCloud;
import Rose.Core.MathUtils;

using namespace RoseEngine;

ParameterBlock<PointCloud> pointCloud;
StructuredBuffer<uint2> sortPairs;
RWTexture2D<float4> outputColor;
uniform float4x4 view;;
uniform float4x4 projection;
uniform uint2    outputExtent;
uniform float    pointSize;

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= outputExtent))
        return;

    const float2 pixelCenter = float2(pixel) + 0.5;

    float4 color = float4(0, 0, 0, 1);

    for (uint i = 0; i < pointCloud.numVertices; i++)
    {
        const uint vertexId = sortPairs[i].y;
        const float3 vertex      = pointCloud.LoadVertex(vertexId);
        const float4 vertexColor = pointCloud.LoadVertexColor(vertexId);

        const float3 viewVertex = mul(view, float4(vertex, 1)).xyz;
        const float4 ndc = mul(projection, float4(viewVertex, 1));

        if (viewVertex.z > 0 || ndc.w == 0 || any(isnan(ndc))) continue;

        const float s = pointSize/2 * max(abs(outputExtent.x*projection[0][0]), abs(outputExtent.y*projection[1][1])) / abs(viewVertex.z);

        const float2 uv = (ndc.xy / ndc.w) * 0.5 + 0.5;
        const float2 dstPixel = uv * outputExtent;
        if (any(pixelCenter < floor(dstPixel - s)) || any(pixelCenter > ceil(dstPixel + s)))
            continue;

        const float4 fragColor = float4(vertexColor.rgb * vertexColor.a, 1 - vertexColor.a);

        color = color*fragColor.a + float4(fragColor.rgb, 0);
    }

    outputColor[pixel] = color;
}
