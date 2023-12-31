#include "lib/common/common.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

#include "lib/encoder.h"

int main(int argc, char** argv) {

    setbuf(stdout, NULL);
    
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_wav_file> [optional_arg]" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* fname_inp = argv[2];
    bool optional_arg = false;

    if (argc > 3) {
        std::string arg(argv[3]);
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        optional_arg = (arg == "true");
    }

    // Initialize whisper context from file
    encoder_context* ctx = encoder_init_from_file(model_path);

    if(ctx == nullptr) {
        std::cerr << "Failed to initialize whisper context from file: " << model_path << std::endl;
        return 1;
    }


    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    if (!::read_wav(fname_inp, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp);
    }

    encoder_full_params eparams = encoder_full_default_params();


    // TODO: Use the context for something

    int res = encoder_full_parallel(
                ctx, 
                eparams, 
                pcmf32.data(), 
                pcmf32.size(),
                1);
    

    std::ostringstream json_output;
    json_output << "{";
    json_output << "\"return_code\": " << (res ? 0 : 1) << ", ";
    json_output << "\"embedding\": [";

    if (res == 0) {
        for (size_t i = 0; i < ctx->state->encoder_embedding.size(); ++i) {
            json_output << ctx->state->encoder_embedding[i];
            if (i < ctx->state->encoder_embedding.size() - 1) {
                json_output << ", ";
            }
        }
    }

    json_output << "]}";
    std::cout << json_output.str() << std::endl;

    return res;
}
