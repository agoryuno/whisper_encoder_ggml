#ifndef ENCODER_STATE_H
#define ENCODER_STATE_H

#define ENCODER_SAMPLE_RATE 16000
#define ENCODER_MAX_SCRATCH_BUFFERS 16
#define ENCODER_N_FFT       400

#define SIN_COS_N_COUNT ENCODER_N_FFT

struct encoder_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct encoder_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct encoder_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode = 0; // number of decoder calls
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    encoder_mel mel;

    // memory buffers used by encode contexts
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[ENCODER_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[ENCODER_MAX_SCRATCH_BUFFERS] = { 0 };

    // std::vector<whisper_segment> result_all;
    // std::vector<whisper_token>   prompt_past;

    mutable std::mt19937 rng; // used for sampling at t > 0.0

    int lang_id = 0; // english by default

    std::string path_model; // populated by encoder_init_from_file()

    std::vector<float> energy; // PCM signal energy

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0; // 0 - use default

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(WHISPER_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(WHISPER_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

#endif // WHISPER_STATE_H