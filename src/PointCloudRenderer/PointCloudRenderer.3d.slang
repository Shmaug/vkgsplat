import Scene.PointCloudScene;
import Rose.Core.MathUtils;

using namespace RoseEngine;

ParameterBlock<PointCloudScene> pointCloud;
StructuredBuffer<uint2> sortPairs;
uniform float4x4 viewProjection;
uniform float3   cameraUp;
uniform float3   cameraRight;
uniform float    pointSize;

struct v2f {
    float4 pos : SV_Position;
    nointerpolation float3 color : COLOR;
};

static const uint3 kQuadMeshIndices[] = {
    uint3(0, 1, 2),
    uint3(1, 3, 2)
};

#define GROUP_SIZE 32
#define TRIS_PER_GROUP ((GROUP_SIZE/4) * 2)

[shader("mesh")]
[outputtopology("triangle")]
[WaveSize(GROUP_SIZE)]
[numthreads(GROUP_SIZE, 1, 1)]
void meshmain(
    in uint3 threadId: SV_DispatchThreadID,
    OutputVertices<v2f, GROUP_SIZE> vertices,
    OutputIndices<uint3, TRIS_PER_GROUP> triangles)
{
    SetMeshOutputCounts(GROUP_SIZE, TRIS_PER_GROUP);

    // groupId * pointsPerGroup
    const uint groupBasePointId = (threadId.x / GROUP_SIZE) * (GROUP_SIZE/4);

    const uint localIdx = threadId.x % GROUP_SIZE;
    const uint idx4     = threadId.x % 4;
    const uint idx2     = threadId.x % 2;

    // 4 verts per quad
    const uint pointId      = threadId.x / 4;
    const uint localPointId = localIdx / 4;

    float3 vertex = FLT_MAX;
    float3 vertexColor = 0;
    if (groupBasePointId + localIdx < pointCloud.numVertices)
    {
        const uint sortedVertexId = sortPairs[groupBasePointId + localIdx].y;
        vertex       = pointCloud.LoadVertex(sortedVertexId);
        vertexColor  = pointCloud.LoadVertexColor(sortedVertexId);
    }
    vertex      = WaveReadLaneAt(vertex,      localIdx / 4);
    vertexColor = WaveReadLaneAt(vertexColor, localIdx / 4);


    const float2 localQuadVert = float2(idx2, idx4/2) * 2 - 1;

    v2f o = {};
    o.pos = mul(viewProjection, float4(vertex + pointSize * (cameraRight * localQuadVert.x + cameraUp * localQuadVert.y), 1));
    o.color = vertexColor;
    vertices[4 * localPointId + idx4] = o;

    if (localIdx < TRIS_PER_GROUP)
    {
        // 2 tris per quad
        const uint triPointId = localIdx / 2;
        triangles[localIdx] = 4 * triPointId + kQuadMeshIndices[idx2];
    }
}

[shader("fragment")]
float4 fsmain(float3 color: COLOR) : SV_Target {
    float alpha = 0.5 + 0.5 * saturate(luminance(color));
    return float4(color * alpha, 1 - alpha);
}
