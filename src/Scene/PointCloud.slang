import Adam.BufferGradient;

struct PointCloud
{
    BufferGradient<3> vertices;
    BufferGradient<4> colors;
    uint numVertices;
};