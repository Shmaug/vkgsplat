import Scene.PointCloud;
import Adam.BufferGradient;

uniform PointCloud pointCloud;
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
float4 evalPoint(const float2 samplePoint, const float3 vertex, const float4 vertexColor)
{
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

    // compute final alpha
    float a = vertexColor.a;
    const float2 d = (samplePoint - screenVertex) / s;
    // a *= 1 - dot(d, d);
    a *= exp(-5 * dot(d, d));

    return float4(vertexColor.rgb * a, 1 - a);
}

[Differentiable]
float4 blend(const float4 dstColor, const float4 srcColor)
{
    return float4(
        dstColor.rgb + srcColor.rgb * dstColor.a,
        dstColor.a * srcColor.a
    );
}

// solves `newColor = blend(dstColor, srcColor)` for dstColor
[Differentiable]
float4 invBlend(const float4 newColor, const float4 srcColor)
{
    const float T = newColor.a / srcColor.a;
    return float4(
        newColor.rgb - srcColor.rgb * T,
        T
    );
}

[Differentiable]
float computeLoss(uint2 pixel, float4 color, no_diff float4 gt) {
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

        const float3 vertex      = pointCloud.vertices.Load(vertexId);
        const float4 vertexColor = pointCloud.colors.Load(vertexId);

        const float4 fragColor = evalPoint(pixelCenter, vertex, vertexColor);

        color = blend(color, fragColor);

        if (color.a <= 1e-6) break;
    }

    outputColor[pixel] = color;
    pixelVertexCounts[pixel] = count;
}

#define BLOCK_DIM 32
#define WAVES_PER_BLOCK ((BLOCK_DIM * BLOCK_DIM)/32)


groupshared uint vertexId;
groupshared uint numRemaining;
groupshared float4 groupReduction[WAVES_PER_BLOCK];

void groupAny(const uint groupThreadId, const bool cond) {
    if (groupThreadId == 0) numRemaining = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(numRemaining, cond ? 1 : 0);
    GroupMemoryBarrierWithGroupSync();    
}

vector<float, N> float4toN<let N : int>(const float4 v) {
    vector<float, N> dst;
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        dst[i] = v[i];
    return dst;
}
float4 floatNto4<let N : int>(const vector<float, N> v) {
    float4 dst;
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        dst[i] = v[i];
    return dst;
}

vector<float, N> BlockLoad<let N : int>(const uint groupThreadId, BufferGradient<N> buf)
{
    // 1st thread loads the data into shared memory
    if (groupThreadId == 0) groupReduction[0] = floatNto4(buf.Load(vertexId));

    GroupMemoryBarrierWithGroupSync();

    // other threads read from shared memory
    return float4toN<N>(groupReduction[0]);
}

void BlockWriteGradient<let N:int>(const uint groupThreadId, BufferGradient<N> dst, vector<float,N> value)
{
    const uint waveIndex = groupThreadId / 32;

    groupAny(groupThreadId, any(value > 0));

    while (numRemaining > 32) {
        // reduce within each wave
        // all waves in the block write their results to shared memory
        value = WaveActiveSum(value);
        if (WaveIsFirstLane())
        {
            groupReduction[waveIndex] = floatNto4(value);
            GroupMemoryBarrierWithGroupSync();
        }
        value = 0;

        if (groupThreadId < WAVES_PER_BLOCK)
            value = float4toN<N>(groupReduction[groupThreadId]);

        groupAny(groupThreadId, any(value > 0));
    }

    if (numRemaining > 0) {
        value = WaveActiveSum(value);

        // 1st thread in block writes results to global memory
        if (groupThreadId == 0) dst.AccumulateGradient(vertexId, value);
    }
}

[shader("compute")]
[numthreads(BLOCK_DIM, BLOCK_DIM, 1)]
[WaveSize(32)]
void __bwd_render(uint3 threadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint2 pixel = threadId.xy;
    const uint flatGroupThreadId = groupThreadId.y * 16 + groupThreadId.x;
    const bool active = all(pixel < outputExtent);

    var color = diffPair(active ? outputColor[pixel] : 0);
    
    const float4 gt = active ? reference[pixel] : 0;
    #if OUTPUT_LOSS
    if (active) outputLoss.InterlockedAddF32(0, computeLoss(pixel, color.p, gt));
    #endif

    __bwd_diff(computeLoss)(pixel, color, gt, 1.0);
    float4 d_color = color.d;

    var pixelCenter = diffPair(float2(pixel) + 0.5);

    const uint count = active ? pixelVertexCounts[pixel] : 0;

    uint maxCount = WaveActiveMax(count);
    if (WaveIsFirstLane()) InterlockedMax(numRemaining, maxCount);
    GroupMemoryBarrierWithGroupSync();
    maxCount = numRemaining;

    for (int i = int(maxCount)-1; i >= 0; i--)
    {
        if (flatGroupThreadId == 0)
            vertexId = sortPairs[i].y;
        GroupMemoryBarrierWithGroupSync();

        groupAny(flatGroupThreadId, i < count);
        if (numRemaining == 0) continue;

        float3 vertex_d = 0;
        float4 vertexColor_d = 0;

        var vertex      = diffPair(BlockLoad(flatGroupThreadId, pointCloud.vertices));
        var vertexColor = diffPair(BlockLoad(flatGroupThreadId, pointCloud.colors));

        if (i < count) {
            var fragColor = diffPair(evalPoint(pixelCenter.p, vertex.p, vertexColor.p));

            // compute color before rendering this vertex
            var inputColor = diffPair(invBlend(color.p, fragColor.p));

            __bwd_diff(blend)(inputColor, fragColor, d_color);
            d_color = inputColor.d;

            __bwd_diff(evalPoint)(pixelCenter, vertex, vertexColor, fragColor.d);

            vertex_d = vertex.d;
            vertexColor_d = vertexColor.d;
        }

        BlockWriteGradient(flatGroupThreadId, pointCloud.vertices, vertex_d);
        BlockWriteGradient(flatGroupThreadId, pointCloud.colors,   vertexColor_d);
    }
}
