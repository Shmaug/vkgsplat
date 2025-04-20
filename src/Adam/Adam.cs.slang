// "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
// https://arxiv.org/pdf/1412.6980

import BufferGradient;

#ifndef NUM_CHANNELS
#define NUM_CHANNELS 1
#endif
static const uint N = NUM_CHANNELS;

static const float kEpsilon = 1e-6;

BufferGradient<N> parameters;

typedef BufferGradient<N>::T T;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint   parameterCount; // number of parameters
    float  stepSize;   // α
    float2 decayRates; // β1, β2
    uint   t; // iteration index
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x;
    
    if (any(index >= parameterCount))
        return;

    T g_t;  // gradient at t w.r.t. parameters at t-1
    T m_t1; // 1st moment at t-1
    T v_t1; // 2nd moment at t-1
    if (t == 0) {
        g_t = 0;
        m_t1 = 0;
    } else {
        g_t  = parameters.LoadGradient(index);
        m_t1 = parameters.LoadMoment1(index);
        v_t1 = parameters.LoadMoment2(index);
    }

    // compute moments at t
    const T m_t = lerp(g_t,       m_t1, decayRates.x);
    const T v_t = lerp(g_t * g_t, v_t1, decayRates.y);

    parameters.StoreMoment1(index, m_t);
    parameters.StoreMoment2(index, v_t);

    const float2 fac = 1 - pow(decayRates, t);
    const float alpha_t = stepSize * sqrt(fac.y) / fac.x;
    
    const T delta = -alpha_t * m_t / (sqrt(v_t) + kEpsilon);

    // compute parameters at t
    if (all(delta == delta) && !any(isnan(delta))) {
        const T prev = parameters.Load(index);
        parameters.Store(index, prev + delta);
    }
}
