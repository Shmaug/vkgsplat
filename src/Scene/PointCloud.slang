void SumReduce<let N : int>(RWByteAddressBuffer dst, const uint addr, vector<float,N> value) {
    [ForceUnroll]
    for (uint i = 0; i < N; i++)
        dst.InterlockedAddF32(addr + i * sizeof(float), value[i]);
}

struct PointCloud
{
    RWByteAddressBuffer vertices;
    RWStructuredBuffer<float4> colors;
    RWByteAddressBuffer vertices_d;
    RWByteAddressBuffer colors_d;
    uint numVertices;

    [BackwardDerivative(LoadVertex_bwd)]
    float3 LoadVertex(uint vertexId) {
        return vertices.Load<float3>(vertexId*sizeof(float3));
    }

    [BackwardDerivative(LoadVertexColor_bwd)]
    float4 LoadVertexColor(uint vertexId) {
        return colors[vertexId];
    }

    void LoadVertex_bwd(uint index, const float3 resultGradient) {
        SumReduce(vertices_d, sizeof(float3) * index, resultGradient);
    }
    void LoadVertexColor_bwd(uint index, const float4 resultGradient) {
        SumReduce(colors_d, sizeof(float4) * index, resultGradient);
    }
};