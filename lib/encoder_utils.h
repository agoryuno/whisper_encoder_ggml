#include <vector>

#include "encoder_state.h"

static bool hann_window(int length, bool periodic, std::vector<float> & output);