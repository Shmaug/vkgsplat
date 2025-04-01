import Scene.GaussianScene;

ParameterBlock<PointCloud> pointCloud;
StructuredBuffer<uint2> sortPairs;
uniform float4x4 viewProjection;
uniform float3   cameraPosition;
uniform float    pointSize;

struct v2f {
    float4 pos : SV_Position;
    float3 color : COLOR;
    float pointSize : SV_PointSize;
};

[shader("vertex")]
v2f vsmain(uint vertexId: SV_VertexID)
{
    const uint sortedVertex = sortPairs[vertexId].y;
    const float3 vertex = pointCloud.LoadVertex(sortedVertex);
    const float3 vertexColor = pointCloud.LoadVertexColor(sortedVertex);

    v2f o;
    o.pointSize = pointSize;
    o.color = vertexColor;
    o.pos = mul(viewProjection, float4(vertex, 1));
    return o;
}

[shader("fragment")]
float4 fsmain(float3 color: COLOR) : SV_Target {
    return float4(color, 0);
}
