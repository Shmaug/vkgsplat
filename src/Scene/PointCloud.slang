struct PointCloud
{
    ByteAddressBuffer vertices;
    StructuredBuffer<float4> colors;
    uint numVertices;

    float3 LoadVertex(uint vertexId)       { return vertices.Load<float3>(vertexId * sizeof(float3)); }
    float4 LoadVertexColor(uint vertexId)  { return colors[vertexId]; }
};