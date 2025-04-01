struct PointCloud
{
    ByteAddressBuffer vertices;
    ByteAddressBuffer colors;

    float3 LoadVertex(uint vertexId)
    {
        return vertices.Load<float3>(vertexId * sizeof(float3));
    }
    
    float3 LoadVertexColor(uint vertexId)
    {
        return colors.Load<float3>(vertexId * sizeof(float3));
    }
};