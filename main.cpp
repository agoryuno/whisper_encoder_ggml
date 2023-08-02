#include <iostream>
#include "lib/whisper.h"

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Initialize whisper context from file
    whisper_context* ctx = whisper_init_from_file_no_state(model_path);

    if(ctx == nullptr) {
        std::cerr << "Failed to initialize whisper context from file: " << model_path << std::endl;
        return 1;
    }

    std::cout << "Successfully initialized whisper context from file: " << model_path << std::endl;

    // TODO: Use the context for something
    
    // Clean up
    // TODO: Release the context if necessary

    return 0;
}
