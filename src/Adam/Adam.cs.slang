// Alg. 1, "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980

typedef float ParameterType;

static const float kEpsilon = 1e-6;

RWStructuredBuffer<ParameterType> parameters;
StructuredBuffer<ParameterType>   gradients; // gradient of parameters

// 1st and 2nd moment vectors.
// size: 2*len(parameters)
RWStructuredBuffer<ParameterType> moments;

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

    // gradient at t w.r.t. parameters at t-1
    const ParameterType g_t  = (t == 0) ? 0 : gradients[i];

    // get moments at t-1
    const ParameterType m_t1 = (t == 0) ? 0 : moments[2*i + 0];
    const ParameterType v_t1 = (t == 0) ? 0 : moments[2*i + 1];

    // compute moments at t
    const ParameterType m_t = lerp(g_t,       m_t1, decayRates.x);
    const ParameterType v_t = lerp(g_t * g_t, v_t1, decayRates.y);

    moments[2*i + 0] = m_t;
    moments[2*i + 1] = v_t;

    /*

    // compute bias-corrected moments at t
    const ParameterType mhat_t = m_t / (1 - pow(decayRates.x, t));
    const ParameterType vhat_t = v_t / (1 - pow(decayRates.y, t));

    // compute parameters at t
    parameters[i] -= stepSize * mhat_t / (sqrt(vhat_t) + kEpsilon);

    /*/

    const float alpha_t = stepSize * sqrt(1 - pow(decayRates.y, t)) / (1 - pow(decayRates.x, t));

    // compute parameters at t
    parameters[i] -= alpha_t * m_t / (sqrt(v_t) + kEpsilon);

    //*/
}
