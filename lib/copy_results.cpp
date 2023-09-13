#include <vector>
#include <cstring> // for memcpy

#include "lib/encoder_state.h"
#include "lib/ggml.h"

// Overload for F32
void set_encoder_result(encoder_state& state, const float* data, size_t size) {
    std::memcpy(state.encoder_embedding.data(), data, size * sizeof(float));
}

// Overload for F16
void set_encoder_result(encoder_state& state, const uint16_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint16_t h = data[i];
        uint32_t sign = ((h >> 15) & 1) << 31;
        uint32_t exponent = ((h >> 10) & 0x1f) << 23;
        uint32_t fraction = (h & 0x3ff) << 23;
        uint32_t f32 = sign | exponent | fraction;
        float f = *reinterpret_cast<float*>(&f32);
        state.encoder_embedding[i] = f;
    }
}

// Overload for double
void set_encoder_result(encoder_state& state, const double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        state.encoder_embedding[i] = static_cast<float>(data[i]);
    }
}

// Overload for I8
void set_encoder_result(encoder_state& state, const int8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        state.encoder_embedding[i] = static_cast<float>(data[i]);
    }
}

// Overload for I16
void set_encoder_result(encoder_state& state, const int16_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        state.encoder_embedding[i] = static_cast<float>(data[i]);
    }
}

// Overload for I32
void set_encoder_result(encoder_state& state, const int32_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        state.encoder_embedding[i] = static_cast<float>(data[i]);
    }
}

// Main function that delegates to the appropriate overload
void set_encoder_result(encoder_state& state, const ggml_tensor& tensor) {
    // Resize the vector
    state.encoder_embedding.resize(tensor.ne[0]);

    switch (tensor.type) {
        case GGML_TYPE_F32:
            set_encoder_result(state, static_cast<const float*>(tensor.data), tensor.ne[0]);
            break;
        case GGML_TYPE_F16:
            set_encoder_result(state, static_cast<const uint16_t*>(tensor.data), tensor.ne[0]);
            break;
        case GGML_TYPE_I8:
            set_encoder_result(state, static_cast<const int8_t*>(tensor.data), tensor.ne[0]);
            break;
        case GGML_TYPE_I16:
            set_encoder_result(state, static_cast<const int16_t*>(tensor.data), tensor.ne[0]);
            break;
        case GGML_TYPE_I32:
            set_encoder_result(state, static_cast<const int32_t*>(tensor.data), tensor.ne[0]);
            break;
        // Add more cases as needed
        default:
            // Handle unsupported types
            break;
    }
}
