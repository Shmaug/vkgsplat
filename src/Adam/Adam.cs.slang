// Alg. 1, "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980

static const float kEpsilon = 1e-6;

RWStructuredBuffer<float> parameters;
StructuredBuffer<float>   gradients; // gradient of parameters
RWStructuredBuffer<float2> moments; // 1st and 2nd moments of gradients

[[vk::push_constant]]
cbuffer PushConstants
{
    uint   parameterCount; // number of parameters
    uint   t; // iteration index
    float2 decayRates; // β1, β2
    float  stepSize;   // α
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;
    
    if (any(i >= parameterCount))
        return;

   
    float g_t; // gradient at t w.r.t. parameters at t-1
    float2 m_t1; // moments at t-1
    if (t == 0) {
        g_t = 0;
        m_t1 = 0;
    } else {
        g_t = gradients[i];
        m_t1 = moments[i];
    }

    // compute moments at t
    const float2 m_t = lerp(float2(g_t, g_t * g_t), m_t1, decayRates);

    moments[i] = m_t;

    /*

    // compute bias-corrected moments at t
    const float2 mhat_t = m_t / (1 - pow(decayRates, t));

    const float dx = stepSize * mhat_t.x / (sqrt(mhat_t.y) + kEpsilon);

    /*/

    const float2 a = 1 - pow(decayRates, t);
    const float alpha_t = stepSize * sqrt(a.y) / a.x;

    // compute parameters at t
    const float dx = alpha_t * m_t.x / (sqrt(m_t.y) + kEpsilon);

    //*/

    // compute parameters at t
    if (all(dx == dx) && !any(isnan(dx)))
        parameters[i] -= dx;

}
