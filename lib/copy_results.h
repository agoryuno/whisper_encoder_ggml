#pragma once
#include <cstddef> // for size_t
#include "encoder_state.h"


void set_encoder_result(encoder_state& state, const float* data, size_t size);

void set_encoder_result(encoder_state& state, const uint16_t* data, size_t size);

void set_encoder_result(encoder_state& state, const double* data, size_t size);

void set_encoder_result(encoder_state& state, const int8_t* data, size_t size);

void set_encoder_result(encoder_state& state, const int16_t* data, size_t size);

void set_encoder_result(encoder_state& state, const int32_t* data, size_t size);

void set_encoder_result(encoder_state& state, const ggml_tensor& tensor);