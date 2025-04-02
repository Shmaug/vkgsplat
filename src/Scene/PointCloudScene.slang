struct PointCloudScene
{
    StructuredBuffer<float4x4> views;
    StructuredBuffer<float4x4> projections;

    ByteAddressBuffer vertices;
    ByteAddressBuffer colors;

    uint numVertices;
    uint numViews;

    float3 LoadVertex(uint vertexId)       { return vertices.Load<float3>(vertexId * sizeof(float3)); }
    float3 LoadVertexColor(uint vertexId)  { return colors  .Load<float3>(vertexId * sizeof(float3)); }
};