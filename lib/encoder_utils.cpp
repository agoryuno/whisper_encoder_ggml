#include <cstdio>
#include <cmath>

#include "encoder_utils.h"

static bool hann_window(int length, bool periodic, std::vector<float> & output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5*(1.0 - cosf((2.0*M_PI*i)/(length + offset)));
    }

    return true;
}