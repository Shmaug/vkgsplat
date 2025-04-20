import Scene.PointCloud;
import Rose.Core.MathUtils;

using namespace RoseEngine;

ParameterBlock<PointCloud> pointCloud;
StructuredBuffer<uint2> sortPairs;
RWTexture2D<float4> outputColor;
RWByteAddressBuffer outputLoss;
Texture2D<float4>   reference;
RWTexture2D<uint>   pixelVertexCounts;
uniform float4x4 view;
uniform float4x4 projection;
uniform uint2    outputExtent;
uniform float    pointSize;


[Differentiable]
float4 evalPoint(const float2 samplePoint, const uint vertexId)
{
    const float3 vertex = pointCloud.LoadVertex(vertexId);

    const float3 viewVertex = mul(view, float4(vertex, 1)).xyz;
    const float4 ndc = mul(projection, float4(viewVertex, 1));

    if (viewVertex.z * projection[2][2] < 0 || ndc.w == 0 || any(isnan(ndc)))
        return float4(0,0,0,1);

    const float s = (pointSize/2) * max(abs(outputExtent.x*projection[0][0]), abs(outputExtent.y*projection[1][1])) / abs(viewVertex.z);

    const float2 uv = (ndc.xy / ndc.w) * 0.5 + 0.5;
    const float2 screenVertex = uv * outputExtent;
    if (any(floor(samplePoint) < floor(screenVertex - s)) || any(ceil(samplePoint) > ceil(screenVertex + s)))
        return float4(0,0,0,1);

    /*
    const float2 clippedMn = max(float2(pixel),     dstPixel - s);
    const float2 clippedMx = min(float2(pixel + 1), dstPixel + s);
    const float area = (clippedMx.y - clippedMn.y) * (clippedMx.x - clippedMn.x);
    */

    float4 vertexColor = pointCloud.LoadVertexColor(vertexId);

    const float2 d = (samplePoint - screenVertex) / s;
    // vertexColor.a *= 1 - dot(d, d);
    vertexColor.a *= exp(-5 * dot(d, d));

    return float4(vertexColor.rgb * vertexColor.a, 1 - vertexColor.a);
}

[Differentiable]
float4 blend(const float4 dstColor, const float4 srcColor)
{
    return float4(
        dstColor.rgb + srcColor.rgb * dstColor.a,
        dstColor.a * srcColor.a
    );
}

// computes input dstColor for blend to produce newColor
[Differentiable]
float4 invBlend(const float4 newColor, const float4 srcColor)
{
    // newColor = blend(dstColor, srcColor)
    // solve for dstColor:
    // newColor.a   = dstColor.a * srcColor.a
    // newColor.rgb = dstColor.rgb + srcColor.rgb * dstColor.a
    // dstColor.a = newColor.a / srcColor.a
    // dstColor.rgb = newColor.rgb - srcColor.rgb * dstColor.a
    const float T = newColor.a / srcColor.a;
    return float4(
        newColor.rgb - srcColor.rgb * T,
        T
    );
}

[Differentiable]
float computeLoss(uint2 pixel, float4 color) {
    float4 gt = no_diff reference[pixel];
    gt.a = 1 - gt.a; // convert alpha to transmittance
    const float4 error = color - gt;
    return dot(error * error, 1.0/4.0);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void render(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= outputExtent))
        return;

    const float2 pixelCenter = float2(pixel) + 0.5;

    float4 color = float4(0, 0, 0, 1);
    uint count = 0;

    while (count < pointCloud.numVertices)
    {
        const uint vertexId = sortPairs[count].y;
        count++;

        const float4 fragColor = evalPoint(pixelCenter, vertexId);

        color = blend(color, fragColor);

        if (color.a <= 1e-6) break;
    }

    #if OUTPUT_LOSS
    outputLoss.InterlockedAddF32(0, computeLoss(pixel, color));
    #endif

    outputColor[pixel] = color;
    pixelVertexCounts[pixel] = count;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void render_bwd(uint3 threadId: SV_DispatchThreadID)
{
    const uint2 pixel = threadId.xy;
    if (any(pixel >= outputExtent))
        return;

    var color = diffPair(outputColor[pixel]);
    
    __bwd_diff(computeLoss)(pixel, color, 1.0);
    float4 d_color = color.d;

    var pixelCenter = diffPair(float2(pixel) + 0.5);

    const int count = pixelVertexCounts[pixel];
    for (int i = count-1; i >= 0; i--)
    {
        const uint vertexId = sortPairs[i].y;

        var fragColor = diffPair(evalPoint(pixelCenter.p, vertexId));

        // compute color before rendering this vertex
        var inputColor = diffPair(invBlend(color.p, fragColor.p));

        __bwd_diff(blend)(inputColor, fragColor, d_color);
        d_color = inputColor.d;

        __bwd_diff(evalPoint)(pixelCenter, vertexId, fragColor.d);
    }
}
