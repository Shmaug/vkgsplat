struct BufferGradient<let N : int> {
    typedef vector<float, N> T;
    static const uint kElementStride = sizeof(float) * N;

    static T _LoadN(RWByteAddressBuffer buf, const uint index) {
        return buf.Load<T>(index * kElementStride);
    }
    static void _StoreN(RWByteAddressBuffer buf, const uint index, const T r) {
        buf.Store<T>(index * kElementStride, r);
    }

    RWByteAddressBuffer data;
    RWByteAddressBuffer gradients;
    RWByteAddressBuffer moments1;
    RWByteAddressBuffer moments2;

    [BackwardDerivative(AccumulateGradient)]
    T Load        (uint index) { return _LoadN(data,      index); }
    T LoadGradient(uint index) { return _LoadN(gradients, index); }
    T LoadMoment1 (uint index) { return _LoadN(moments1,  index); }
    T LoadMoment2 (uint index) { return _LoadN(moments2,  index); }
    
    void Store       (uint index, const T value) { return _StoreN(data,     index, value); }
    void StoreMoment1(uint index, const T value) { return _StoreN(moments1, index, value); }
    void StoreMoment2(uint index, const T value) { return _StoreN(moments2, index, value); }
        
    void AccumulateGradient(uint index, const T resultGradient) {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            gradients.InterlockedAddF32((index * N + i) * sizeof(float), resultGradient[i]);
    }
};
