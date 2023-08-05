#ifndef ENCODER_H
#define ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef WHISPER_SHARED
#    ifdef _WIN32
#        ifdef WHISPER_BUILD
#            define WHISPER_API __declspec(dllexport)
#        else
#            define WHISPER_API __declspec(dllimport)
#        endif
#    else
#        define WHISPER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WHISPER_API
#endif

typedef void (*encoder_log_callback)(const char * line);

typedef struct encoder_model_loader {
    void * context;
    size_t (*read)(void * ctx, void * output, size_t read_size);
    bool    (*eof)(void * ctx);
    void  (*close)(void * ctx);
} encoder_model_loader;

struct encoder_context;

// This function will properly delete an instance of encoder_context
void encoder_context_destroy(encoder_context* ctx);

WHISPER_API struct encoder_context * encoder_init_from_file_no_state(const char * path_model);

#endif