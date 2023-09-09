#ifndef ENCODER_H
#define ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "encoder_state.h"

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



struct encoder_context;
struct encoder_full_params;



typedef struct encoder_model_loader {
    void * context;
    size_t (*read)(void * ctx, void * output, size_t read_size);
    bool    (*eof)(void * ctx);
    void  (*close)(void * ctx);
} encoder_model_loader;

// This function will properly delete an instance of encoder_context
void encoder_context_destroy(encoder_context* ctx);

WHISPER_API struct encoder_context * encoder_init_from_file_no_state(const char * path_model);

// Parameters for the encoder_full() function
// If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
// encoder_full_default_params()
struct encoder_full_params {
    int offset_ms;          // start offset in ms
    int duration_ms;        // audio duration to process in ms
    int n_threads;

    bool single_segment;    // force single segment output (useful for streaming)
    bool print_progress;    // print progress information

    // [EXPERIMENTAL] speed-up techniques
    // note: these can significantly reduce the quality of the output
    bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
    bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
    int  audio_ctx;         // overwrite the audio context size (0 = use default)

    // [EXPERIMENTAL] [TDRZ] tinydiarize
    bool tdrz_enable;       // enable tinydiarize speaker turn detection

    // for auto-detection, set to nullptr, "" or "auto"
    const char * language;
    bool detect_language;

    void * encoder_begin_callback_user_data;

};

WHISPER_API struct encoder_full_params encoder_full_default_params();

typedef void (*encoder_log_callback)(const char * line);
WHISPER_API void encoder_set_log_callback(encoder_log_callback callback);

int encoder_full_with_state(
        struct encoder_context * ctx,
          struct encoder_state * state,
    struct encoder_full_params   params,
                   const float * samples,
                           int   n_samples);


int encoder_full_parallel(
        struct encoder_context * ctx,
        struct encoder_full_params params,
        const float * samples,
        int n_samples,
        int n_processors);

struct encoder_context * encoder_init_from_file(const char * path_model);

#endif