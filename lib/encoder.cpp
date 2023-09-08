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


// available whisper models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

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


// default hparams (Whisper tiny)
struct encoder_hparams {
    int32_t n_mels        = 80;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t ftype         = 1;
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
    if (params.audio_ctx > encoder_n_audio_ctx(ctx);) {
        log("%s: audio_ctx is larger than the maximum allowed (%d > %d)\n", __func__, params.audio_ctx, encoder_n_audio_ctx(ctx););
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
        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads)) {
            log("%s: failed to encode\n", __func__);
            return -6;
        }

        // if there is a very short audio segment left to process, we remove 
        // any past prompt since it tends
        // to confuse the decoder and often make it repeat or hallucinate stuff
        if (seek > seek_start && seek + 500 >= seek_end) {
            prompt_past.clear();
        }

        int best_decoder_id = 0;

        for (int it = 0; it < (int) temperatures.size(); ++it) {
            const float t_cur = temperatures[it];

            int n_decoders_cur = 1;

            switch (params.strategy) {
                case whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY:
                    {
                        if (t_cur > 0.0f) {
                            n_decoders_cur = params.greedy.best_of;
                        }
                    } break;
                case whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH:
                    {
                        if (t_cur > 0.0f) {
                            n_decoders_cur = params.greedy.best_of;
                        } else {
                            n_decoders_cur = params.beam_search.beam_size;
                        }
                    } break;
            };

            n_decoders_cur = std::max(1, n_decoders_cur);

            WHISPER_PRINT_DEBUG("\n%s: decoding with %d decoders, temperature = %.2f\n", __func__, n_decoders_cur, t_cur);

            // TAGS: WHISPER_DECODER_INIT
            for (int j = 0; j < n_decoders_cur; ++j) {
                auto & decoder = state->decoders[j];

                decoder.kv_self.n = 0;

                decoder.sequence.tokens.clear();
                decoder.sequence.result_len       = 0;
                decoder.sequence.sum_logprobs_all = 0.0;
                decoder.sequence.sum_logprobs     = -INFINITY;
                decoder.sequence.avg_logprobs     = -INFINITY;
                decoder.sequence.entropy          = 0.0;
                decoder.sequence.score            = -INFINITY;

                decoder.seek_delta = 100*WHISPER_CHUNK_SIZE;

                decoder.failed    = false;
                decoder.completed = false;
                decoder.has_ts    = false;
            }

            // init prompt and kv cache for the current iteration
            // run whisper_decoder() only for decoder 0 and copy the results for the other decoders
            {
                prompt.clear();

                // if we have already generated some text, use it as a prompt to condition the next generation
                if (!prompt_past.empty() && t_cur < 0.5f && params.n_max_text_ctx > 0) {
                    int n_take = std::min(std::min(params.n_max_text_ctx, whisper_n_text_ctx(ctx)/2), int(prompt_past.size()));

                    prompt = { whisper_token_prev(ctx) };
                    prompt.insert(prompt.begin() + 1, prompt_past.end() - n_take, prompt_past.end());
                }

                // init new transcription with sot, language (opt) and task tokens
                prompt.insert(prompt.end(), prompt_init.begin(), prompt_init.end());

                // print the prompt
                WHISPER_PRINT_DEBUG("\n\n");
                for (int i = 0; i < (int) prompt.size(); i++) {
                    WHISPER_PRINT_DEBUG("%s: prompt[%d] = %s\n", __func__, i, ctx->vocab.id_to_token.at(prompt[i]).c_str());
                }
                WHISPER_PRINT_DEBUG("\n\n");

                if (!whisper_decode_internal(*ctx, *state, state->decoders[0], prompt.data(), prompt.size(), 0, params.n_threads)) {
                    log("%s: failed to decode\n", __func__);
                    return -7;
                }

                {
                    const int64_t t_start_sample_us = ggml_time_us();

                    whisper_process_logits(*ctx, *state, params, state->decoders[0], t_cur);

                    state->decoders[0].kv_self.n += prompt.size();

                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        memcpy(decoder.kv_self.k->data, state->decoders[0].kv_self.k->data, ggml_nbytes(decoder.kv_self.k));
                        memcpy(decoder.kv_self.v->data, state->decoders[0].kv_self.v->data, ggml_nbytes(decoder.kv_self.v));

                        decoder.kv_self.n += prompt.size();

                        memcpy(decoder.probs.data(), state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(), state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }

                    state->t_sample_us += ggml_time_us() - t_start_sample_us;
                }
            }

            for (int i = 0, n_max = whisper_n_text_ctx(ctx)/2 - 4; i < n_max; ++i) {
                const int64_t t_start_sample_us = ggml_time_us();

                // store the KV caches of all decoders when doing beam-search
                if (params.strategy == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH) {
                    kv_bufs.resize(n_decoders_cur);
                    for (int j = 0; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        if (decoder.completed || decoder.failed) {
                            continue;
                        }

                        kv_bufs[j].k.resize(ggml_nbytes(decoder.kv_self.k));
                        kv_bufs[j].v.resize(ggml_nbytes(decoder.kv_self.v));

                        memcpy(kv_bufs[j].k.data(), decoder.kv_self.k->data, kv_bufs[j].k.size());
                        memcpy(kv_bufs[j].v.data(), decoder.kv_self.v->data, kv_bufs[j].v.size());
                    }

                    beam_candidates.clear();
                }

                // generate new sequence candidates for each decoder
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.completed || decoder.failed) {
                        continue;
                    }

                    switch (params.strategy) {
                        case whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY:
                            {
                                if (t_cur < 1e-6f) {
                                    decoder.sequence.tokens.push_back(whisper_sample_token(*ctx, *state, decoder, true));
                                } else {
                                    decoder.sequence.tokens.push_back(whisper_sample_token(*ctx, *state, decoder, false));
                                }

                                decoder.sequence.sum_logprobs_all += decoder.sequence.tokens.back().plog;
                            } break;
                        case whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH:
                            {
                                const auto tokens_new = whisper_sample_token_topk(*ctx, *state, decoder, params.beam_search.beam_size);

                                for (const auto & token : tokens_new) {
                                    beam_candidates.push_back({ j, decoder.seek_delta, decoder.has_ts, decoder.sequence });
                                    beam_candidates.back().sequence.tokens.push_back(token);
                                    beam_candidates.back().sequence.sum_logprobs_all += token.plog;

                                    //WHISPER_PRINT_DEBUG("%s: beam candidate: %s (%f, %f)\n", __func__, ctx->vocab.id_to_token.at(token.id).c_str(), token.plog, beam_candidates.back().sequence.sum_logprobs_all);
                                }
                            } break;
                    };
                }

                // for beam-search, choose the top candidates and update the KV caches
                if (params.strategy == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH) {
                    std::sort(
                            beam_candidates.begin(),
                            beam_candidates.end(),
                            [](const beam_candidate & a, const beam_candidate & b) {
                        return a.sequence.sum_logprobs_all > b.sequence.sum_logprobs_all;
                    });

                    uint32_t cur_c = 0;

                    for (int j = 0; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        if (decoder.completed || decoder.failed) {
                            continue;
                        }

                        auto & cur = beam_candidates[cur_c++];

                        while (beam_candidates.size() > cur_c && beam_candidates[cur_c].sequence.sum_logprobs_all == cur.sequence.sum_logprobs_all && i > 0) {
                            ++cur_c;
                        }

                        decoder.sequence   = cur.sequence;
                        decoder.seek_delta = cur.seek_delta;
                        decoder.has_ts     = cur.has_ts;

                        memcpy(decoder.kv_self.k->data, kv_bufs[cur.decoder_idx].k.data(), kv_bufs[cur.decoder_idx].k.size());
                        memcpy(decoder.kv_self.v->data, kv_bufs[cur.decoder_idx].v.data(), kv_bufs[cur.decoder_idx].v.size());

                        WHISPER_PRINT_DEBUG("%s: beam search: decoder %d: from decoder %d: token = %10s, plog = %8.5f, sum_logprobs = %8.5f\n",
                                __func__, j, cur.decoder_idx, ctx->vocab.id_to_token.at(decoder.sequence.tokens.back().id).c_str(), decoder.sequence.tokens.back().plog, decoder.sequence.sum_logprobs_all);
                    }
                }

                // update the decoder state
                // - check if the sequence is completed
                // - check if the sequence is failed
                // - update sliding window based on timestamp tokens
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.completed || decoder.failed) {
                        continue;
                    }

                    auto & has_ts     = decoder.has_ts;
                    auto & failed     = decoder.failed;
                    auto & completed  = decoder.completed;
                    auto & seek_delta = decoder.seek_delta;
                    auto & result_len = decoder.sequence.result_len;

                    {
                        const auto & token = decoder.sequence.tokens.back();

                        // timestamp token - update sliding window
                        if (token.id > whisper_token_beg(ctx)) {
                            const int seek_delta_new = 2*(token.id - whisper_token_beg(ctx));

                            // do not allow to go back in time
                            if (has_ts && seek_delta > seek_delta_new && result_len < i) {
                                failed = true; // TODO: maybe this is not a failure ?
                                continue;
                            }

                            seek_delta = seek_delta_new;
                            result_len = i + 1;
                            has_ts = true;
                        }

#ifdef WHISPER_DEBUG
                        {
                            const auto tt = token.pt > 0.10 ? ctx->vocab.id_to_token.at(token.tid) : "[?]";
                            WHISPER_PRINT_DEBUG("%s: id = %3d, decoder = %d, token = %6d, p = %6.3f, ts = %10s, %6.3f, result_len = %4d '%s'\n",
                                    __func__, i, j, token.id, token.p, tt.c_str(), token.pt, result_len, ctx->vocab.id_to_token.at(token.id).c_str());
                        }
#endif

                        // end of segment
                        if (token.id == whisper_token_eot(ctx) ||               // end of text token
                           (params.max_tokens > 0 && i >= params.max_tokens) || // max tokens per segment reached
                           (has_ts && seek + seek_delta + 100 >= seek_end)      // end of audio reached
                           ) {
                            if (result_len == 0) {
                                if (seek + seek_delta + 100 >= seek_end) {
                                    result_len = i + 1;
                                } else {
                                    failed = true;
                                    continue;
                                }
                            }

                            if (params.single_segment) {
                                result_len = i + 1;
                                seek_delta = 100*WHISPER_CHUNK_SIZE;
                            }

                            completed = true;
                            continue;
                        }

                        // TESTS: if no tensors are loaded, it means we are running tests
                        if (ctx->model.n_loaded == 0) {
                            seek_delta = 100*WHISPER_CHUNK_SIZE;
                            completed = true;
                            continue;
                        }
                    }

                    // sometimes, the decoding can get stuck in a repetition loop
                    // this is an attempt to mitigate such cases - we flag the decoding as failed and use a fallback strategy
                    if (i == n_max - 1 && (result_len == 0 || seek_delta < 100*WHISPER_CHUNK_SIZE/2)) {
                        failed = true;
                        continue;
                    }
                }

                // check if all decoders have finished (i.e. completed or failed)
                {
                    bool completed_all = true;

                    for (int j = 0; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        if (decoder.completed || decoder.failed) {
                            continue;
                        }

                        completed_all = false;
                    }

                    if (completed_all) {
                        break;
                    }
                }

                state->t_sample_us += ggml_time_us() - t_start_sample_us;

                // obtain logits for the next token
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.failed || decoder.completed) {
                        continue;
                    }

                    decoder.tokens_tmp.resize(1);
                    decoder.tokens_tmp[0] = decoder.sequence.tokens.back().id;

                    //WHISPER_PRINT_DEBUG("%s: decoder %d: token %d, kv_self.n %d, seek_delta %d\n", __func__, j, decoder.tokens_tmp[0], decoder.kv_self.n, decoder.seek_delta);

                    if (!whisper_decode_internal(*ctx, *state, decoder, decoder.tokens_tmp.data(), decoder.tokens_tmp.size(), decoder.kv_self.n, params.n_threads)) {
                        log("%s: failed to decode\n", __func__);
                        return -8;
                    }

                    {
                        const int64_t t_start_sample_us = ggml_time_us();

                        whisper_process_logits(*ctx, *state, params, decoder, t_cur);

                        ++decoder.kv_self.n;

                        state->t_sample_us += ggml_time_us() - t_start_sample_us;
                    }
                }
            }

            // rank the resulting sequences and select the best one
            {
                double best_score = -INFINITY;

                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.failed) {
                        continue;
                    }

                    decoder.sequence.tokens.resize(decoder.sequence.result_len);
                    whisper_sequence_score(params, decoder.sequence);

                    WHISPER_PRINT_DEBUG("%s: decoder %2d: score = %8.5f, result_len = %3d, avg_logprobs = %8.5f, entropy = %8.5f\n",
                            __func__, j, decoder.sequence.score, decoder.sequence.result_len, decoder.sequence.avg_logprobs, decoder.sequence.entropy);

                    if (decoder.sequence.result_len > 32 && decoder.sequence.entropy < params.entropy_thold) {
                        WHISPER_PRINT_DEBUG("%s: decoder %2d: failed due to entropy %8.5f < %8.5f\n",
                                __func__, j, decoder.sequence.entropy, params.entropy_thold);

                        decoder.failed = true;
                        state->n_fail_h++;

                        continue;
                    }

                    if (best_score < decoder.sequence.score) {
                        best_score = decoder.sequence.score;
                        best_decoder_id = j;
                    }
                }

                WHISPER_PRINT_DEBUG("%s: best decoder = %d\n", __func__, best_decoder_id);
            }

            // was the decoding successful for the current temperature?
            // do fallback only if:
            // - we are not at the last temperature
            // - we are not at the end of the audio (3 sec)
            if (it != (int) temperatures.size() - 1 &&
                seek_end - seek > 10*WHISPER_CHUNK_SIZE) {
                bool success = true;

                const auto & decoder = state->decoders[best_decoder_id];

                if (decoder.failed || decoder.sequence.avg_logprobs < params.logprob_thold) {
                    success = false;
                    state->n_fail_p++;
                }

                if (success) {
                    //for (auto & token : ctx->decoders[best_decoder_id].sequence.tokens) {
                    //    WHISPER_PRINT_DEBUG("%s: token = %d, p = %6.3f, pt = %6.3f, ts = %s, str = %s\n", __func__, token.id, token.p, token.pt, ctx->vocab.id_to_token.at(token.tid).c_str(), ctx->vocab.id_to_token.at(token.id).c_str());
                    //}

                    break;
                }
            }

            WHISPER_PRINT_DEBUG("\n%s: failed to decode with temperature = %.2f\n", __func__, t_cur);
        }

        // output results through a user-provided callback
        {
            const auto & best_decoder = state->decoders[best_decoder_id];

            const auto seek_delta = best_decoder.seek_delta;
            const auto result_len = best_decoder.sequence.result_len;

            const auto & tokens_cur = best_decoder.sequence.tokens;

            //WHISPER_PRINT_DEBUG("prompt_init.size() = %d, prompt.size() = %d, result_len = %d, seek_delta = %d\n", prompt_init.size(), prompt.size(), result_len, seek_delta);

            // update prompt_past
            prompt_past.clear();
            if (prompt.front() == whisper_token_prev(ctx)) {
                prompt_past.insert(prompt_past.end(), prompt.begin() + 1, prompt.end() - prompt_init.size());
            }

            for (int i = 0; i < result_len; ++i) {
                prompt_past.push_back(tokens_cur[i].id);
            }

            if (!tokens_cur.empty() && ctx->model.n_loaded > 0) {
                int  i0 = 0;
                auto t0 = seek + 2*(tokens_cur.front().tid - whisper_token_beg(ctx));

                std::string text;
                bool speaker_turn_next = false;

                for (int i = 0; i < (int) tokens_cur.size(); i++) {
                    //printf("%s: %18s %6.3f %18s %6.3f\n", __func__,
                    //        ctx->vocab.id_to_token[tokens_cur[i].id].c_str(), tokens_cur[i].p,
                    //        ctx->vocab.id_to_token[tokens_cur[i].tid].c_str(), tokens_cur[i].pt);

                    if (params.print_special || tokens_cur[i].id < whisper_token_eot(ctx)) {
                        text += whisper_token_to_str(ctx, tokens_cur[i].id);
                    }

                    // [TDRZ] record if speaker turn was predicted after current segment
                    if (params.tdrz_enable && tokens_cur[i].id == whisper_token_solm(ctx)) {
                        speaker_turn_next = true;
                    }

                    if (tokens_cur[i].id > whisper_token_beg(ctx) && !params.single_segment) {
                        const auto t1 = seek + 2*(tokens_cur[i].tid - whisper_token_beg(ctx));

                        if (!text.empty()) {
                            const auto tt0 = params.speed_up ? 2*t0 : t0;
                            const auto tt1 = params.speed_up ? 2*t1 : t1;

                            if (params.print_realtime) {
                                if (params.print_timestamps) {
                                    printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                                } else {
                                    printf("%s", text.c_str());
                                    fflush(stdout);
                                }
                            }

                            //printf("tt0 = %d, tt1 = %d, text = %s, token = %s, token_id = %d, tid = %d\n", tt0, tt1, text.c_str(), ctx->vocab.id_to_token[tokens_cur[i].id].c_str(), tokens_cur[i].id, tokens_cur[i].tid);

                            result_all.push_back({ tt0, tt1, text, {}, speaker_turn_next });
                            for (int j = i0; j <= i; j++) {
                                result_all.back().tokens.push_back(tokens_cur[j]);
                            }

                            int n_new = 1;

                            if (params.token_timestamps) {
                                whisper_exp_compute_token_level_timestamps(
                                        *ctx, *state, result_all.size() - 1, params.thold_pt, params.thold_ptsum);

                                if (params.max_len > 0) {
                                    n_new = whisper_wrap_segment(*ctx, *state, params.max_len, params.split_on_word);
                                }
                            }
                            if (params.new_segment_callback) {
                                params.new_segment_callback(ctx, state, n_new, params.new_segment_callback_user_data);
                            }
                        }
                        text = "";
                        while (i < (int) tokens_cur.size() && tokens_cur[i].id > whisper_token_beg(ctx)) {
                            i++;
                        }
                        i--;
                        t0 = t1;
                        i0 = i + 1;
                        speaker_turn_next = false;
                    }
                }

                if (!text.empty()) {
                    const auto t1 = seek + seek_delta;

                    const auto tt0 = params.speed_up ? 2*t0 : t0;
                    const auto tt1 = params.speed_up ? 2*t1 : t1;

                    if (params.print_realtime) {
                        if (params.print_timestamps) {
                            printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                        } else {
                            printf("%s", text.c_str());
                            fflush(stdout);
                        }
                    }

                    result_all.push_back({ tt0, tt1, text, {} , speaker_turn_next });
                    for (int j = i0; j < (int) tokens_cur.size(); j++) {
                        result_all.back().tokens.push_back(tokens_cur[j]);
                    }

                    int n_new = 1;

                    if (params.token_timestamps) {
                        whisper_exp_compute_token_level_timestamps(
                                *ctx, *state, result_all.size() - 1, params.thold_pt, params.thold_ptsum);

                        if (params.max_len > 0) {
                            n_new = whisper_wrap_segment(*ctx, *state, params.max_len, params.split_on_word);
                        }
                    }
                    if (params.new_segment_callback) {
                        params.new_segment_callback(ctx, state, n_new, params.new_segment_callback_user_data);
                    }
                }
            }

            // update audio window
            seek += seek_delta;

            WHISPER_PRINT_DEBUG("seek = %d, seek_delta = %d\n", seek, seek_delta);
        }
    }

    return 0;
}