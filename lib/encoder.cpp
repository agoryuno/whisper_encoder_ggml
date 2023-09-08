#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <random>

#include "ggml.h"
#include "encoder.h"
#include "encoder_state.h"
#include "encoder_utils.h"


#if defined(GGML_BIG_ENDIAN)
#include <bit>

template<typename T>
static T byteswap(T value) {
    return std::byteswap(value);
}

template<>
float byteswap(float value) {
    return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template<typename T>
static void byteswap_tensor_data(ggml_tensor * tensor) {
    T * datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_I16: {
            byteswap_tensor_data<int16_t>(tensor);
            break;
        }
        case GGML_TYPE_F16: {
            byteswap_tensor_data<ggml_fp16_t>(tensor);
            break;
        }
        case GGML_TYPE_I32: {
            byteswap_tensor_data<int32_t>(tensor);
            break;
        }
        case GGML_TYPE_F32: {
            byteswap_tensor_data<float>(tensor);
            break;
        }
        default: { // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)            \
    do {                              \
        for (auto & datum : f.data) { \
            datum = byteswap(datum);  \
        }                             \
    } while (0)
#define BYTESWAP_TENSOR(t)       \
    do {                         \
        byteswap_tensor(tensor); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#endif

#define WHISPER_ASSERT(x) \
    do { \
        if (!(x)) { \
            log("WHISPER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// define this to enable verbose trace logging - useful for debugging purposes
//#define WHISPER_DEBUG

#if defined(WHISPER_DEBUG)
#define WHISPER_PRINT_DEBUG(...) \
    do { \
        fprintf(stderr, __VA_ARGS__); \
    } while (0)
#else
#define WHISPER_PRINT_DEBUG(...)
#endif

//#define WHISPER_USE_FLASH_ATTN
//#define WHISPER_USE_FLASH_FF
#define WHISPER_MAX_DECODERS 16

#define WHISPER_USE_SCRATCH




static const size_t MB = 1ull*1024*1024;

static const std::map<e_model, size_t> MEM_REQ_SCRATCH0 = {
    { MODEL_TINY,     62ull*MB },
    { MODEL_BASE,     80ull*MB },
    { MODEL_SMALL,   120ull*MB },
    { MODEL_MEDIUM,  158ull*MB },
    { MODEL_LARGE,   198ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH1 = {
    { MODEL_TINY,     18ull*MB },
    { MODEL_BASE,     24ull*MB },
    { MODEL_SMALL,    36ull*MB },
    { MODEL_MEDIUM,   48ull*MB },
    { MODEL_LARGE,    60ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH2 = {
    { MODEL_TINY,      4ull*MB },
    { MODEL_BASE,      4ull*MB },
    { MODEL_SMALL,     6ull*MB },
    { MODEL_MEDIUM,    7ull*MB },
    { MODEL_LARGE,     9ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH3 = {
    { MODEL_TINY,      4ull*MB },
    { MODEL_BASE,      4ull*MB },
    { MODEL_SMALL,     6ull*MB },
    { MODEL_MEDIUM,    7ull*MB },
    { MODEL_LARGE,     9ull*MB },
};

static const std::map<ggml_type, std::map<e_model, size_t>> MEM_REQ_MODEL = {
    { GGML_TYPE_F32,
        {
            { MODEL_TINY,     74ull*MB },
            { MODEL_BASE,    142ull*MB },
            { MODEL_SMALL,   466ull*MB },
            { MODEL_MEDIUM, 1464ull*MB },
            { MODEL_LARGE,  2952ull*MB },
        },
    },
    { GGML_TYPE_F16,
        {
            { MODEL_TINY,     74ull*MB },
            { MODEL_BASE,    142ull*MB },
            { MODEL_SMALL,   466ull*MB },
            { MODEL_MEDIUM, 1464ull*MB },
            { MODEL_LARGE,  2952ull*MB },
        },
    },
    { GGML_TYPE_Q4_0,
        {
            { MODEL_TINY,     26ull*MB },
            { MODEL_BASE,     50ull*MB },
            { MODEL_SMALL,   154ull*MB },
            { MODEL_MEDIUM,  470ull*MB },
            { MODEL_LARGE,   940ull*MB },
        },
    },
    { GGML_TYPE_Q4_1,
        {
            { MODEL_TINY,     32ull*MB },
            { MODEL_BASE,     58ull*MB },
            { MODEL_SMALL,   182ull*MB },
            { MODEL_MEDIUM,  562ull*MB },
            { MODEL_LARGE,  1124ull*MB },
        },
    },
    { GGML_TYPE_Q5_0,
        {
            { MODEL_TINY,     30ull*MB },
            { MODEL_BASE,     54ull*MB },
            { MODEL_SMALL,   170ull*MB },
            { MODEL_MEDIUM,  516ull*MB },
            { MODEL_LARGE,  1034ull*MB },
        },
    },
    { GGML_TYPE_Q5_1,
        {
            { MODEL_TINY,     32ull*MB },
            { MODEL_BASE,     58ull*MB },
            { MODEL_SMALL,   182ull*MB },
            { MODEL_MEDIUM,  562ull*MB },
            { MODEL_LARGE,  1124ull*MB },
        },
    },
    { GGML_TYPE_Q8_0,
        {
            { MODEL_TINY,     45ull*MB },
            { MODEL_BASE,     84ull*MB },
            { MODEL_SMALL,   268ull*MB },
            { MODEL_MEDIUM,  834ull*MB },
            { MODEL_LARGE,  1674ull*MB },
        },
    },
};

static const std::map<e_model, size_t> MEM_REQ_KV_SELF = {
    { MODEL_TINY,      3ull*MB },
    { MODEL_BASE,      6ull*MB },
    { MODEL_SMALL,    16ull*MB },
    { MODEL_MEDIUM,   43ull*MB },
    { MODEL_LARGE,    71ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_KV_CROSS = {
    { MODEL_TINY,      9ull*MB },
    { MODEL_BASE,     18ull*MB },
    { MODEL_SMALL,    53ull*MB },
    { MODEL_MEDIUM,  141ull*MB },
    { MODEL_LARGE,   235ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_ENCODE = {
    { MODEL_TINY,     30ull*MB },
    { MODEL_BASE,     38ull*MB },
    { MODEL_SMALL,    56ull*MB },
    { MODEL_MEDIUM,   74ull*MB },
    { MODEL_LARGE,    94ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_DECODE = {
    { MODEL_TINY,      3ull*MB },
    { MODEL_BASE,      5ull*MB },
    { MODEL_SMALL,    10ull*MB },
    { MODEL_MEDIUM,   18ull*MB },
    { MODEL_LARGE,    27ull*MB },
};

static void whisper_default_log(const char * text) {
    fprintf(stderr, "%s", text);
}

static whisper_log_callback whisper_log = whisper_default_log;

static void log(const char * fmt, ...) {
    if (!whisper_log) return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    whisper_log(buf);
}


struct encoder_full_params encoder_full_default_params() {
    struct encoder_full_params result = {
        /* offset_ms */      0,
        /* duration_ms */    0,
        /*.n_threads */      std::min(4, (int32_t) std::thread::hardware_concurrency()),
        /* single_segment */ false,
        /* print_progress */ false,

        /* speed_up */       false,
        /* debug_mode */     false,
        /* audio_ctx */      0,
    };
    return result;
};


template<typename T>
static void read_safe(encoder_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

void encoder_context_destroy(encoder_context* ctx) {
    // Perform any necessary cleanup of ctx members
    delete ctx;
}

// evaluate the encoder with the given state
//
// given audio recording (more specifically, its log mel spectrogram), runs forward pass of the encoder
// part of the transformer model and returns the encoded features
//
//   - wctx:      the model
//   - wstate:     the state of the encoder
//   - n_threads:  number of threads to use
//   - mel_offset: offset in the mel spectrogram (i.e. audio offset)
//
static bool encode_internal(
        encoder_context & wctx,
          encoder_state & wstate,
              const int   mel_offset,
              const int   n_threads) {

    const int64_t t_start_us = ggml_time_us();

    const auto & model   = wctx.model;
    const auto & mel_inp = wstate.mel;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;

    const int n_mels = hparams.n_mels;
    assert(mel_inp.n_mel == n_mels);

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.buf_compute.size(),
        /*.mem_buffer =*/ wstate.buf_compute.data(),
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    wstate.use_buf(ctx0, 0);

    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    assert(mel->type == GGML_TYPE_F32);
    {
        float * dst = (float *) mel->data;
        memset(dst, 0, ggml_nbytes(mel));

        const int i0 = std::min(mel_offset, mel_inp.n_len);
        const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);

        for (int j = 0; j < mel_inp.n_mel; ++j) {
            for (int i = i0; i < i1; ++i) {
                dst[j*2*n_ctx + (i - i0)] = mel_inp.data[j*mel_inp.n_len + i];
            }
        }
    }

    struct ggml_tensor * cur;

#ifndef WHISPER_USE_COREML
    const bool use_coreml = false;
#else
    const bool use_coreml = wstate.ctx_coreml != nullptr;
#endif

#ifndef WHISPER_USE_OPENVINO
    const bool use_openvino = false;
#else
    const bool use_openvino = wstate.ctx_openvino != nullptr;
#endif

    if (!use_coreml && !use_openvino) {
        // convolution + gelu
        {
            wstate.use_buf(ctx0, 1);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0,
                        model.e_conv_1_b,
                        cur),
                    cur);

            cur = ggml_gelu(ctx0, cur);

            wstate.use_buf(ctx0, 0);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0,
                        model.e_conv_2_b,
                        cur),
                    cur);

            cur = ggml_gelu(ctx0, cur);
        }

        wstate.use_buf(ctx0, 3);

        // ===================================================================
        // NOTE: experimenting with partial evaluation of the encoder (ignore)
        //static int iter = -1;
        //const int n_iter = 1500/n_ctx;

        //iter = (iter + 1) % n_iter;

        //if (iter == 0) {
        //    memset(model.memory_cross_k->data, 0, ggml_nbytes(model.memory_cross_k));
        //    memset(model.memory_cross_v->data, 0, ggml_nbytes(model.memory_cross_v));
        //}

        static int iter = 0;

        const size_t e_pe_stride = model.e_pe->ne[0]*ggml_element_size(model.e_pe);
        const size_t e_pe_offset = model.e_pe->ne[0]*ggml_element_size(model.e_pe)*n_ctx*iter;

        struct ggml_tensor * e_pe = ggml_view_2d(ctx0, model.e_pe, model.e_pe->ne[0], n_ctx, e_pe_stride, e_pe_offset);

        cur = ggml_add(ctx0, e_pe, ggml_transpose(ctx0, cur));

        // ===================================================================

        // original:
        //cur = ggml_add(ctx0, model.e_pe, ggml_transpose(ctx0, cur));

        struct ggml_tensor * inpL = cur;

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers_encoder[il];

            // norm
            {
                wstate.use_buf(ctx0, 0);

                cur = ggml_norm(ctx0, inpL, hparams.eps);

                // cur = ln_0_w*cur + ln_0_b
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, layer.attn_ln_0_w, cur),
                            cur),
                        ggml_repeat(ctx0, layer.attn_ln_0_b, cur));
            }

            // self-attention
            {
                wstate.use_buf(ctx0, 1);

                struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                        layer.attn_q_w,
                        cur);

                Qcur = ggml_add(ctx0,
                        ggml_repeat(ctx0,
                            layer.attn_q_b,
                            Qcur),
                        Qcur);

                //Qcur = ggml_scale_inplace(ctx0, Qcur, ggml_new_f32(ctx0, pow(float(n_state)/n_head, -0.25)));

                // note: no bias for Key
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                        layer.attn_k_w,
                        cur);

                //Kcur = ggml_scale_inplace(ctx0, Kcur, ggml_new_f32(ctx0, pow(float(n_state)/n_head, -0.25)));

                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                        layer.attn_v_w,
                        cur);

                Vcur = ggml_add(ctx0,
                        ggml_repeat(ctx0,
                            layer.attn_v_b,
                            Vcur),
                        Vcur);

                // ------

                wstate.use_buf(ctx0, 0);

#ifdef WHISPER_USE_FLASH_ATTN
                struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                            0, 2, 1, 3);

                struct ggml_tensor * K =
                    ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Kcur,
                                ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                            0, 2, 1, 3);

                struct ggml_tensor * V =
                    ggml_cpy(ctx0,
                            ggml_permute(ctx0,
                                ggml_reshape_3d(ctx0,
                                    Vcur,
                                    n_state/n_head, n_head, n_ctx),
                                1, 2, 0, 3),
                            ggml_new_tensor_3d(ctx0, wctx.itype, n_ctx, n_state/n_head, n_head));

                struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, false);
#else
                struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state/n_head, n_head, n_ctx)),
                            0, 2, 1, 3);

                struct ggml_tensor * K =
                    ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Kcur,
                                ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                            0, 2, 1, 3);

                // K * Q
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                struct ggml_tensor * KQ_scaled =
                    ggml_scale_inplace(ctx0,
                            KQ,
                            ggml_new_f32(ctx0, 1.0f/sqrt(float(n_state)/n_head))
                            );

                struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

                struct ggml_tensor * V =
                    ggml_cpy(ctx0,
                            ggml_permute(ctx0,
                                ggml_reshape_3d(ctx0,
                                    Vcur,
                                    n_state/n_head, n_head, n_ctx),
                                1, 2, 0, 3),
                            ggml_new_tensor_3d(ctx0, wctx.itype, n_ctx, n_state/n_head, n_head)
                            );

                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
#endif
                struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                wstate.use_buf(ctx0, 1);

                cur = ggml_cpy(ctx0,
                        KQV_merged,
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));
            }

            // projection
            {
                wstate.use_buf(ctx0, 0);

                cur = ggml_mul_mat(ctx0,
                        layer.attn_ln_1_w,
                        cur);

                wstate.use_buf(ctx0, 1);

                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, layer.attn_ln_1_b, cur),
                        cur);
            }

            wstate.use_buf(ctx0, 2);

            // add the input
            cur = ggml_add(ctx0, cur, inpL);

            struct ggml_tensor * inpFF = cur;

            // feed-forward network
            {
                // norm
                {
                    wstate.use_buf(ctx0, 0);

                    cur = ggml_norm(ctx0, inpFF, hparams.eps);

                    wstate.use_buf(ctx0, 1);

                    // cur = mlp_ln_w*cur + mlp_ln_b
                    cur = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                ggml_repeat(ctx0, layer.mlp_ln_w, cur),
                                cur),
                            ggml_repeat(ctx0, layer.mlp_ln_b, cur));
                }

#ifdef WHISPER_USE_FLASH_FF
                wstate.use_buf(ctx0, 0);

                cur = ggml_flash_ff(ctx0,
                        ggml_cpy(ctx0, cur, ggml_new_tensor_2d(ctx0, wstate.itype, n_state, n_ctx)),
                        layer.mlp_0_w, layer.mlp_0_b, layer.mlp_1_w, layer.mlp_1_b);
#else
                wstate.use_buf(ctx0, 0);

                // fully connected
                cur = ggml_mul_mat(ctx0,
                        layer.mlp_0_w,
                        cur);

                wstate.use_buf(ctx0, 1);

                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, layer.mlp_0_b, cur),
                        cur);

                wstate.use_buf(ctx0, 0);

                // GELU activation
                cur = ggml_gelu(ctx0, cur);

                wstate.use_buf(ctx0, 1);

                // projection
                cur = ggml_mul_mat(ctx0,
                        layer.mlp_1_w,
                        cur);

                wstate.use_buf(ctx0, 0);

                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, layer.mlp_1_b, cur),
                        cur);
#endif
            }

            wstate.use_buf(ctx0, 3);

            inpL = ggml_add(ctx0, cur, inpFF);
        }

        cur = inpL;

        // norm
        {
            wstate.use_buf(ctx0, 0);

            cur = ggml_norm(ctx0, cur, hparams.eps);

            wstate.use_buf(ctx0, 1);

            // cur = ln_f_g*cur + ln_f_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.e_ln_w, cur),
                        cur),
                    ggml_repeat(ctx0, model.e_ln_b, cur));
        }

        wstate.use_buf(ctx0, -1);

        // run the computation
        {
            struct ggml_cgraph gf = {};

            ggml_build_forward_expand  (&gf, cur);
            ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

            //ggml_graph_print(&gf);
        }
    }
#ifdef WHISPER_USE_COREML
    else if (use_coreml) {
        wstate.use_buf(ctx0, -1);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);

        whisper_coreml_encode(wstate.ctx_coreml, (float *) mel->data, (float *) cur->data);
    }
#endif
#ifdef WHISPER_USE_OPENVINO
    else if (use_openvino) {
        wstate.use_buf(ctx0, -1);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);

        if (!whisper_openvino_encode(wstate.ctx_openvino, mel, cur)) {
            return false;
        }
    }
#endif

    // cur
    //{
    //    printf("ne0 = %d\n", cur->ne[0]);
    //    printf("ne1 = %d\n", cur->ne[1]);
    //    for (int i = 0; i < 10; ++i) {
    //        printf("%8.4f ", ((float *)(cur->data))[i]);
    //    }
    //    printf("... ");
    //    for (int i = cur->ne[0] - 10; i < cur->ne[0]; ++i) {
    //        printf("%8.4f ", ((float *)(cur->data))[i]);
    //    }
    //    printf("\n");
    //}

    // pre-compute cross-attention memory
    /*
    {
        struct ggml_cgraph gf = {};

        // TODO: hack to disconnect the encoded features from the previous graph
        cur->op = GGML_OP_NONE;
        cur->src[0] = nullptr;
        cur->src[1] = nullptr;

        for (int il = 0; il < model.hparams.n_text_layer; ++il) {
            auto& layer = model.layers_decoder[il];

            wstate.use_buf(ctx0, 0);

            struct ggml_tensor* Kcross = ggml_mul_mat(ctx0,
                layer.cross_attn_k_w,
                cur);

            Kcross = ggml_scale_inplace(ctx0, Kcross, ggml_new_f32(ctx0, pow(float(n_state) / n_head, -0.25)));

            wstate.use_buf(ctx0, 1);

            struct ggml_tensor* Vcross = ggml_mul_mat(ctx0,
                layer.cross_attn_v_w,
                cur);

            Vcross = ggml_add(ctx0,
                ggml_repeat(ctx0,
                    layer.cross_attn_v_b,
                    Vcross),
                Vcross);

            wstate.use_buf(ctx0, -1);

            Vcross = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcross, n_state, n_ctx));

            struct ggml_tensor * k = ggml_view_1d(ctx0, wstate.kv_cross.k, n_state*n_ctx, (ggml_element_size(wstate.kv_cross.k)*n_state)*(il*n_ctx));
            struct ggml_tensor * v = ggml_view_2d(ctx0, wstate.kv_cross.v, n_ctx, n_state,
                    (   n_ctx)*ggml_element_size(wstate.kv_cross.v),
                    (il*n_ctx)*ggml_element_size(wstate.kv_cross.v)*n_state);

            ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcross, k));
            ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcross, v));
        }

        ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
        //ggml_graph_print(&gf);
    }
    */

    ////////////////////////////////////////////////////////////////////////////

    //printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //        ggml_used_mem(ctx0)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(0)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(1)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(2)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(3)/1024.0/1024.0);

    ggml_free(ctx0);

    wstate.t_encode_us += ggml_time_us() - t_start_us;
    wstate.n_encode++;

    return true;
}

// load the model from a ggml file
//
// file format:
//
//   - hparams
//   - pre-computed mel filters
//   - vocab
//   - weights
//
// see the convert-pt-to-ggml.py script for details
//

static bool encoder_model_load(struct encoder_model_loader * loader, 
                               encoder_context & wctx) {
    log("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            log("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    {
        auto & hparams = model.hparams;

        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        if (hparams.n_audio_layer == 4) {
            model.type = e_model::MODEL_TINY;
        }

        if (hparams.n_audio_layer == 6) {
            model.type = e_model::MODEL_BASE;
        }

        if (hparams.n_audio_layer == 12) {
            model.type = e_model::MODEL_SMALL;
        }

        if (hparams.n_audio_layer == 24) {
            model.type = e_model::MODEL_MEDIUM;
        }

        if (hparams.n_audio_layer == 32) {
            model.type = e_model::MODEL_LARGE;
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
        
        #ifdef DEBUG_MODE
            printf("`ggml_ftype` = %d\n", wctx.wtype);
        #endif

        if (wctx.wtype == GGML_TYPE_COUNT) {
            log("%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        const size_t scale = model.hparams.ftype ? 1 : 2;

        log("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        log("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        log("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        log("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        log("%s: n_mels        = %d\n", __func__, hparams.n_mels);
        log("%s: ftype         = %d\n", __func__, model.hparams.ftype);
        log("%s: qntvr         = %d\n", __func__, qntvr);
        log("%s: type          = %d\n", __func__, model.type);

        // print memory requirements
        {
            // this is the total memory required to run the inference
            const size_t mem_required =
                     MEM_REQ_SCRATCH0.at(model.type) +
                     MEM_REQ_SCRATCH1.at(model.type) +
                     MEM_REQ_SCRATCH2.at(model.type) +
                     MEM_REQ_SCRATCH3.at(model.type) +
                scale*MEM_REQ_MODEL.at(wctx.wtype).at(model.type) +
                scale*MEM_REQ_KV_CROSS.at(model.type) +
                scale*std::max(MEM_REQ_ENCODE.at(model.type), MEM_REQ_DECODE.at(model.type));

            // this is the memory required by one decoder
            const size_t mem_required_decoder =
                scale*MEM_REQ_KV_SELF.at(model.type);

            log("%s: mem required  = %7.2f MB (+ %7.2f MB per decoder)\n", __func__,
                    mem_required / 1024.0 / 1024.0, mem_required_decoder / 1024.0 / 1024.0);
        }

        // initialize all memory buffers
        // always have at least one decoder

        wctx.model.buf = new std::vector<uint8_t>();
        wctx.model.buf->resize(scale*MEM_REQ_MODEL.at(wctx.wtype).at(model.type));

        #ifdef DEBUG_MODE
            printf("MODEL SIZE:\n");
            printf("\t`scale` = %zu\n", scale);
            printf("\t`model.type` = %d\n", model.type);
            printf("\t`wctx.model.buf.size()` = %zu\n", wctx.model.buf->size());
        #endif
        // we skip initialization of the state until it is needed
        // because it might be that state will always be provided externally.
    }

    // load mel filters
    {
        auto & filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    size_t ctx_size = 0;

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

    {
        const auto & hparams = model.hparams;

        // const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx   = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;
        const int n_mels = hparams.n_mels;

        // encoder

        {
            ctx_size += n_audio_ctx*n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_pe;

            ctx_size += 3*n_mels*n_audio_state*ggml_type_sizef(vtype);         // e_conv_1_w
            ctx_size +=          n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_conv_1_b

            ctx_size += 3*n_audio_state*n_audio_state*ggml_type_sizef(vtype);         // e_conv_2_w
            ctx_size +=                 n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_conv_2_b

            ctx_size += n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_ln_w;
            ctx_size += n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_ln_b;
        }


        // encoder layers
        {
            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_w
            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_b

            ctx_size += n_audio_layer*(4*n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // mlp_0_w
            ctx_size += n_audio_layer*(              4*n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_0_b

            ctx_size += n_audio_layer*(4*n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // mlp_1_w
            ctx_size += n_audio_layer*(                n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_1_b

            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_w
            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_b

            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_q_w
            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_q_b

            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype)); // attn_k_w

            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_v_w
            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_v_b

            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_ln_1_w
            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_1_b
        }

        ctx_size += (15 + 15*n_audio_layer)*512; 

        log("%s: model ctx     = %7.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    
    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ wctx.model.buf->size(),
            /*.mem_buffer =*/ wctx.model.buf->data(),
            /*.no_alloc   =*/ false,
        };

        #ifdef DEBUG_MODE
            printf("`params.mem_size` = %zu\n", params.mem_size);
        #endif

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            log("%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        auto & ctx = model.ctx;

        const auto & hparams = model.hparams;

        // const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx   = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_mels = hparams.n_mels;

        model.layers_encoder.resize(n_audio_layer);
        // model.layers_decoder.resize(n_text_layer);
        
        #ifdef DEBUG_MODE
            printf("`%s`: starting encoder conversion.\n", __func__);
        #endif

        // encoder
        {
            #ifdef DEBUG_MODE
                printf("creating ggml tensor for PE\n");
            #endif
            model.e_pe       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);

            #ifdef DEBUG_MODE
                printf("creating ggml tensors for first Conv layer\n");
            #endif
            model.e_conv_1_w = ggml_new_tensor_3d(ctx, vtype,         3, n_mels, n_audio_state);
            model.e_conv_1_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

            #ifdef DEBUG_MODE
                printf("creating ggml tensors for second Conv layer\n");
            #endif
            model.e_conv_2_w = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
            model.e_conv_2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

            #ifdef DEBUG_MODE
                printf("creating ggml tensors for LayerNorm\n");
            #endif
            model.e_ln_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
            model.e_ln_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

            // map by name
            model.tensors["encoder.positional_embedding"] = model.e_pe;

            model.tensors["encoder.conv1.weight"]         = model.e_conv_1_w;
            model.tensors["encoder.conv1.bias"]           = model.e_conv_1_b;

            model.tensors["encoder.conv2.weight"]         = model.e_conv_2_w;
            model.tensors["encoder.conv2.bias"]           = model.e_conv_2_b;

            model.tensors["encoder.ln_post.weight"]       = model.e_ln_w;
            model.tensors["encoder.ln_post.bias"]         = model.e_ln_b;

            for (int i = 0; i < n_audio_layer; ++i) {
                #ifdef DEBUG_MODE
                    printf("#################################\n");
                    printf("creating ggml tensors for layer %d\n", i);
                    printf("#################################\n");
                #endif
                auto & layer = model.layers_encoder[i];

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for LN layer\n");
                #endif
                layer.mlp_ln_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.mlp_ln_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for MLP layer 1\n");
                #endif
                layer.mlp_0_w     = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, 4*n_audio_state);
                layer.mlp_0_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for MLP layer 2\n");
                #endif
                layer.mlp_1_w     = ggml_new_tensor_2d(ctx, wtype,         4*n_audio_state, n_audio_state);
                layer.mlp_1_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for attn LN layer 1\n");
                #endif
                layer.attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for attn Q\n");
                #endif
                layer.attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for attn K\n");
                #endif
                layer.attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for attn V\n");
                #endif
                layer.attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                #ifdef DEBUG_MODE
                    printf("creating ggml tensors for LN 2\n");
                #endif
                layer.attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                // map by name
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]     = layer.mlp_ln_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]       = layer.mlp_ln_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.weight"]      = layer.mlp_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.bias"]        = layer.mlp_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.weight"]      = layer.mlp_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.bias"]        = layer.mlp_1_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.weight"]    = layer.attn_ln_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.bias"]      = layer.attn_ln_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.weight"] = layer.attn_q_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.bias"]   = layer.attn_q_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.key.weight"]   = layer.attn_k_w;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.weight"] = layer.attn_v_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.bias"]   = layer.attn_v_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.weight"]   = layer.attn_ln_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.bias"]     = layer.attn_ln_1_b;
            }
        }


    // load weights
    {
        size_t total_size = 0;

        model.n_loaded = 0;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                log("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                log("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                log("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                        __func__, ne[0], ne[1], ne[2], (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                log("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                log("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
            BYTESWAP_TENSOR(tensor);

            //printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype), ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        log("%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);

        if (model.n_loaded == 0) {
            log("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            log("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}
}


struct encoder_context * encoder_init_no_state(struct encoder_model_loader * loader) {
    ggml_time_init();

    #ifdef DEBUG_MODE
        printf("`%s` called.\n", __func__);
    #endif

    encoder_context * ctx = new encoder_context;

    if (!encoder_model_load(loader, *ctx)) {
        loader->close(loader->context);
        log("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    return ctx;
}


struct encoder_context * encoder_init_from_file_no_state(const char * path_model) {

    log("%s: loading model from '%s'\n", __func__, path_model);

    auto fin = std::ifstream(path_model, std::ios::binary);
    if (!fin) {
        log("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    encoder_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    auto ctx = encoder_init_no_state(&loader);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
};


int encoder_full_with_state(
        struct encoder_context * ctx,
          struct encoder_state * state,
    struct encoder_full_params   params,
                   const float * samples,
                           int   n_samples) {
    // clear old results
    //auto & result_all = state->result_all;

    //result_all.clear();

    if (n_samples > 0) {
        // compute log mel spectrogram
        if (params.speed_up) {
            // TODO: Replace PV with more advanced algorithm
            log("%s: failed to compute log mel spectrogram\n", __func__);
            return -1;
        } else {
            if (whisper_pcm_to_mel_with_state(
                            ctx, 
                            state, 
                            samples, 
                            n_samples, 
                            params.n_threads) != 0) {
                log("%s: failed to compute log mel spectrogram\n", __func__);
                return -2;
            }
        }
    }

    /*if (params.token_timestamps) {
        state->t_beg    = 0;
        state->t_last   = 0;
        state->tid_last = 0;
        if (n_samples > 0) {
            state->energy = get_signal_energy(samples, n_samples, 32);
        }
    }*/

    const int seek_start = params.offset_ms/10;
    const int seek_end = params.duration_ms == 0 ? encoder_n_len_from_state(state) : seek_start + params.duration_ms/10;

    // if length of spectrogram is less than 1.0s (100 frames), then return
    // basically don't process anything that is less than 1.0s
    // see issue #39: https://github.com/ggerganov/whisper.cpp/issues/39
    if (seek_end < seek_start + (params.speed_up ? 50 : 100)) {
        return 0;
    }

    // a set of temperatures to use
    // [ t0, t0 + delta, t0 + 2*delta, ..., < 1.0f + 1e-6f ]
    /* std::vector<float> temperatures;
    if (params.temperature_inc > 0.0f) {
        for (float t = params.temperature; t < 1.0f + 1e-6f; t += params.temperature_inc) {
            temperatures.push_back(t);
        }
    } else {
        temperatures.push_back(params.temperature);
    }*/

    // initialize the decoders
    /*int n_decoders = 1;

    switch (params.strategy) {
        case WHISPER_SAMPLING_GREEDY:
            {
                n_decoders = params.greedy.best_of;
            } break;
        case WHISPER_SAMPLING_BEAM_SEARCH:
            {
                n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
            } break;
    };

    n_decoders = std::max(1, n_decoders);

    // TAGS: WHISPER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        if (decoder.kv_self.ctx == nullptr) {
            decoder.kv_self = state->decoders[0].kv_self;
            if (!kv_cache_reinit(decoder.kv_self)) {
                log("%s: kv_cache_reinit() failed for self-attention, decoder %d\n", __func__, j);
                return -4;
            }

            WHISPER_PRINT_DEBUG("%s: initialized self-attention kv cache, decoder %d\n", __func__, j);

            decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

            decoder.probs.resize   (ctx->vocab.n_vocab);
            decoder.logits.resize  (ctx->vocab.n_vocab);
            decoder.logprobs.resize(ctx->vocab.n_vocab);
        }
    }

    // the accumulated text context so far
    auto & prompt_past = state->prompt_past;
    if (params.no_context) {
        prompt_past.clear();
    }

    // prepare prompt
    {
        std::vector<whisper_token> prompt_tokens;

        // initial prompt
        if (!params.prompt_tokens && params.initial_prompt) {
            prompt_tokens.resize(1024);
            prompt_tokens.resize(whisper_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size()));
            params.prompt_tokens   = prompt_tokens.data();
            params.prompt_n_tokens = prompt_tokens.size();
        }

        // prepend the prompt tokens to the prompt_past
        if (params.prompt_tokens && params.prompt_n_tokens > 0) {
            // parse tokens from the pointer
            for (int i = 0; i < params.prompt_n_tokens; i++) {
                prompt_past.push_back(params.prompt_tokens[i]);
            }
            std::rotate(prompt_past.begin(), prompt_past.end() - params.prompt_n_tokens, prompt_past.end());
        }
    }
    */

    // overwrite audio_ctx, max allowed is hparams.n_audio_ctx
    if (params.audio_ctx > encoder_n_audio_ctx(ctx)) {
        log("%s: audio_ctx is larger than the maximum allowed (%d > %d)\n", 
                __func__, 
                params.audio_ctx, 
                encoder_n_audio_ctx(ctx));
        return -5;
    }
    state->exp_n_audio_ctx = params.audio_ctx;

    // these tokens determine the task that will be performed
    /*
    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx) };
    if (whisper_is_multilingual(ctx)) {
        const int lang_id = whisper_lang_id(params.language);
        state->lang_id = lang_id;
        prompt_init.push_back(whisper_token_lang(ctx, lang_id));
        if (params.translate) {
            prompt_init.push_back(whisper_token_translate(ctx));
        } else {
            prompt_init.push_back(whisper_token_transcribe(ctx));
        }
    }
    */

    int seek = seek_start;

    /*
    std::vector<whisper_token> prompt;
    prompt.reserve(whisper_n_text_ctx(ctx));
    */

    // beam-search helpers
    /*
    struct kv_buf {
        std::vector<uint8_t> k;
        std::vector<uint8_t> v;
    };

    std::vector<kv_buf> kv_bufs;

    struct beam_candidate {
        int decoder_idx;
        int seek_delta;

        bool has_ts;

        whisper_sequence sequence;
    };

    std::vector<beam_candidate> beam_candidates;
    */

    // main loop
    while (true) {
        /*if (params.progress_callback) {
            const int progress_cur = (100*(seek - seek_start))/(seek_end - seek_start);

            params.progress_callback(
                ctx, ctx->state, progress_cur, params.progress_callback_user_data);
        }*/

        // of only 1 second left, then stop
        if (seek + 100 >= seek_end) {
            break;
        }

        /*
        if (params.encoder_begin_callback) {
            if (params.encoder_begin_callback(ctx, state, params.encoder_begin_callback_user_data) == false) {
                log("%s: encoder_begin_callback returned false - aborting\n", __func__);
                break;
            }
        }
        */

        // encode audio features starting at offset seek
        if (!encode_internal(*ctx, *state, seek, params.n_threads)) {
            log("%s: failed to encode\n", __func__);
            return -6;
        }

    }        

    return 0;

}


struct encoder_state * encoder_init_state(encoder_context * ctx) {
    fill_sin_cos_table();
    encoder_state * state = new encoder_state;

    const size_t scale = ctx->model.hparams.ftype ? 1 : 2;

    if (!kv_cache_init(ctx->model.hparams, scale * MEM_REQ_KV_SELF.at(ctx->model.type), state->decoders[0].kv_self, ctx->itype, ctx->model.hparams.n_text_ctx)) {
        log("%s: kv_cache_init() failed for self-attention cache\n", __func__);
        delete state;
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->decoders[0].kv_self.k) + ggml_nbytes(state->decoders[0].kv_self.v);
        log("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

    if (!kv_cache_init(ctx->model.hparams, scale * MEM_REQ_KV_CROSS.at(ctx->model.type), state->kv_cross, ctx->itype, ctx->model.hparams.n_audio_ctx)) {
        log("%s: kv_cache_init() failed for cross-attention cache\n", __func__);
        delete state;
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_cross.k) + ggml_nbytes(state->kv_cross.v);
        log("%s: kv cross size = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

#ifdef WHISPER_USE_COREML
    const auto path_coreml = whisper_get_coreml_path_encoder(ctx->path_model);

    log("%s: loading Core ML model from '%s'\n", __func__, path_coreml.c_str());
    log("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_coreml = whisper_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        log("%s: failed to load Core ML model from '%s'\n", __func__, path_coreml.c_str());
#ifndef WHISPER_COREML_ALLOW_FALLBACK
        return nullptr;
#endif
    } else {
        log("%s: Core ML model loaded\n", __func__);
    }
#endif

    state->logits.reserve(ctx->vocab.n_vocab * ctx->model.hparams.n_text_ctx);

    state->logits_id.reserve(ctx->model.hparams.n_vocab);

    // TAGS: WHISPER_DECODER_INIT
    state->decoders[0].sequence.tokens.reserve(ctx->model.hparams.n_text_ctx);

    state->decoders[0].probs.reserve(ctx->vocab.n_vocab);
    state->decoders[0].logits.reserve(ctx->vocab.n_vocab);
    state->decoders[0].logprobs.reserve(ctx->vocab.n_vocab);
    state->buf_compute.resize(scale * std::max(MEM_REQ_ENCODE.at(ctx->model.type), MEM_REQ_DECODE.at(ctx->model.type)));

    state->buf_scratch[0].resize(MEM_REQ_SCRATCH0.at(ctx->model.type));
    state->buf_scratch[1].resize(MEM_REQ_SCRATCH1.at(ctx->model.type));
    state->buf_scratch[2].resize(MEM_REQ_SCRATCH2.at(ctx->model.type));
    state->buf_scratch[3].resize(MEM_REQ_SCRATCH3.at(ctx->model.type));

    state->rng = std::mt19937(0);

    return state;
}

int encoder_full(
        struct encoder_context * ctx,
    struct encoder_full_params   params,
                   const float * samples,
                           int   n_samples) {
    return encoder_full_with_state(ctx, ctx->state, params, samples, n_samples);
}

int encoder_full_parallel(
        struct encoder_context * ctx,
        struct encoder_full_params params,
        const float * samples,
        int n_samples,
        int n_processors) {
    if (n_processors == 1) {
        return encoder_full(ctx, params, samples, n_samples);
    }
    int ret = 0;

    // prepare separate states for each thread
    std::vector<encoder_state*> states;

    const int offset_samples = (ENCODER_SAMPLE_RATE*params.offset_ms)/1000;
    const int n_samples_per_processor = (n_samples - offset_samples)/n_processors;

    // the calling thread will process the first chunk
    // while the other threads will process the remaining chunks

    std::vector<std::thread> workers(n_processors - 1);
    for (int i = 0; i < n_processors - 1; ++i) {
        // create a new state for each thread
        states.push_back(encoder_init_state(ctx));

        const int start_samples = offset_samples + (i + 1)*n_samples_per_processor;
        const int n_samples_cur = (i == n_processors - 2) ? n_samples - start_samples : n_samples_per_processor;

        auto params_cur = params;

        params_cur.offset_ms = 0;
        params_cur.print_progress = false;
        params_cur.print_realtime = false;

        params_cur.new_segment_callback = nullptr;
        params_cur.new_segment_callback_user_data = nullptr;

        params_cur.progress_callback = nullptr;
        params_cur.progress_callback_user_data = nullptr;

        workers[i] = std::thread(whisper_full_with_state, ctx, states[i], std::move(params_cur), samples + start_samples, n_samples_cur);
    }

    {
        auto params_cur = params;

        // We need to disable the print real-time for this one as well, otherwise it will show only for the first chunk.
        params_cur.print_realtime = false;

        // Run the first transformation using default state but only for the first chunk.
        ret = whisper_full_with_state(ctx, ctx->state, std::move(params_cur), samples, offset_samples + n_samples_per_processor);
    }

    for (int i = 0; i < n_processors - 1; ++i) {
        workers[i].join();
    }

    const int64_t offset_t = (int64_t) params.offset_ms/10.0;

    // combine results into result_state->result_all from all other states
    for (int i = 0; i < n_processors - 1; ++i) {
        auto& results_i = states[i]->result_all;

        for (auto& result : results_i) {
            // correct the segment timestamp taking into account the offset
            result.t0 += 100 * ((i + 1) * n_samples_per_processor) / WHISPER_SAMPLE_RATE + offset_t;
            result.t1 += 100 * ((i + 1) * n_samples_per_processor) / WHISPER_SAMPLE_RATE + offset_t;

            // make sure that segments are not overlapping
            if (!ctx->state->result_all.empty()) {
                result.t0 = std::max(result.t0, ctx->state->result_all.back().t1);
            }

            ctx->state->result_all.push_back(std::move(result));

            // call the new_segment_callback for each segment
            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, ctx->state, 1, params.new_segment_callback_user_data);
            }
        }

        ctx->state->t_mel_us += states[i]->t_mel_us;

        ctx->state->t_sample_us += states[i]->t_sample_us;
        ctx->state->t_encode_us += states[i]->t_encode_us;
        ctx->state->t_decode_us += states[i]->t_decode_us;

        whisper_free_state(states[i]);
    }

    // average the timings
    ctx->state->t_mel_us    /= n_processors;
    ctx->state->t_sample_us /= n_processors;
    ctx->state->t_encode_us /= n_processors;
    ctx->state->t_decode_us /= n_processors;

    // print information about the audio boundaries
    log("\n");
    log("%s: the audio has been split into %d chunks at the following times:\n", __func__, n_processors);
    for (int i = 0; i < n_processors - 1; ++i) {
        log("%s: split %d - %s\n", __func__, (i + 1), to_timestamp(100*((i + 1)*n_samples_per_processor)/WHISPER_SAMPLE_RATE + offset_t).c_str());
    }
    log("%s: the transcription quality may be degraded near these boundaries\n", __func__);

    return ret;
}