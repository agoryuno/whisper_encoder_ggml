#include <vector>


#include "../ggml.h"
#include "encoder_state.h"

static bool hann_window(int length, bool periodic, std::vector<float> & output);

int encoder_n_len_from_state(struct encoder_state * state);

typedef void (*encoder_log_callback)(const char * line);

int encoder_n_audio_ctx(struct encoder_context * ctx);

static std::vector<float> get_signal_energy(
                const float * signal, 
                int n_samples, 
                int n_samples_per_half_window);


static void log_mel_spectrogram_worker_thread(
                    int ith, 
                    const std::vector<float> & hann, 
                    const std::vector<float> & samples,
                    int n_samples, 
                    int frame_size, int frame_step, int n_threads,
                    const encoder_filters & filters, 
                    encoder_mel & mel);


int whisper_pcm_to_mel_with_state(
            struct encoder_context * ctx, 
            struct encoder_state * state, 
            const float * samples, 
            int n_samples, 
            int n_threads);

static bool log_mel_spectrogram(
              encoder_state & estate,
              const float * samples,
              const int   n_samples,
              const int   /*sample_rate*/,
              const int   frame_size,
              const int   frame_step,
              const int   n_mel,
              const int   n_threads,
              const encoder_filters & filters,
              const bool   debug,
              encoder_mel & mel);