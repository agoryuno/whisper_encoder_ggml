#ifndef ENCODER_STATE_H
#define ENCODER_STATE_H

#define ENCODER_SAMPLE_RATE 16000
#define ENCODER_MAX_SCRATCH_BUFFERS 16
#define ENCODER_N_FFT       400
#define ENCODER_HOP_LENGTH  160
#define ENCODER_N_MEL       80

#define SIN_COS_N_COUNT ENCODER_N_FFT

#define WHISPER_USE_SCRATCH
#define WHISPER_MAX_SCRATCH_BUFFERS 16

#include <map>
#include <string>
#include <random>
#include "ggml.h"

struct encoder_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};


// audio encoding layer
struct encoder_layer {
    // encoder.blocks.*.attn_ln
    struct ggml_tensor * attn_ln_0_w;
    struct ggml_tensor * attn_ln_0_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor * attn_ln_1_w;
    struct ggml_tensor * attn_ln_1_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor * mlp_0_w;
    struct ggml_tensor * mlp_0_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;
};


// available whisper models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};


// default hparams (Whisper tiny)
struct encoder_hparams {
    int32_t n_mels        = 80;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t ftype         = 1;
    float   eps           = 1e-5f;
};


struct encoder_model {
    e_model type = MODEL_UNKNOWN;

    encoder_hparams hparams;
    encoder_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor * e_pe;

    // encoder.conv1
    struct ggml_tensor * e_conv_1_w;
    struct ggml_tensor * e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor * e_conv_2_w;
    struct ggml_tensor * e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor * e_ln_w;
    struct ggml_tensor * e_ln_b;

    // decoder.positional_embedding
    //struct ggml_tensor * d_pe;

    // decoder.token_embedding
    //struct ggml_tensor * d_te;

    // decoder.ln
    //struct ggml_tensor * d_ln_w;
    //struct ggml_tensor * d_ln_b;

    std::vector<encoder_layer> layers_encoder;

    // context
    struct ggml_context * ctx;

    // the model memory buffer is read-only and can be shared between processors
    std::vector<uint8_t> * buf;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
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

    // cross-attention KV cache for the decoders
    // shared between all decoders
    //whisper_kv_cache kv_cross;
    encoder_mel mel;

    //whisper_decoder decoders[WHISPER_MAX_DECODERS] = {};

    // memory buffers used by encode contexts
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[ENCODER_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[ENCODER_MAX_SCRATCH_BUFFERS] = { 0 };

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    //std::vector<float> logits;

    // std::vector<whisper_segment> result_all;
    // std::vector<whisper_token>   prompt_past;

    // work container used to avoid memory allocations
    //std::vector<std::pair<double, whisper_vocab::id>> logits_id;

    mutable std::mt19937 rng; // used for sampling at t > 0.0

    //int lang_id = 0; // english by default

    std::string path_model; // populated by encoder_init_from_file()
#ifdef WHISPER_USE_COREML
    whisper_coreml_context * ctx_coreml = nullptr;
#endif

#ifdef WHISPER_USE_OPENVINO
    whisper_openvino_context * ctx_openvino = nullptr;
#endif

    // [EXPERIMENTAL] token-level timestamps data
    //int64_t t_beg = 0;
    //int64_t t_last = 0;
    //whisper_token tid_last;
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


struct encoder_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16; // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16; // intermediate type (FP32 or FP16)

    encoder_model model;
    encoder_state * state = nullptr;

    std::string path_model; // populated by whisper_init_from_file()
};

#endif // WHISPER_STATE_H