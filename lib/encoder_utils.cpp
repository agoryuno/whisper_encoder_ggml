#include <algorithm>
#include <cstdio>
#include <cmath>
#include <thread>
#include <fstream>
#include <cassert>
#include <string>
#include <cstring>

#include "ggml.h"
#include "encoder_utils.h"


int encoder_n_len_from_state(struct encoder_state * state) {
    return state->mel.n_len_org;
}

static void encoder_default_log(const char * text) {
    fprintf(stderr, "%s", text);
}

int encoder_n_audio_ctx(struct encoder_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

static float sin_vals[SIN_COS_N_COUNT];
static float cos_vals[SIN_COS_N_COUNT];

static encoder_log_callback encoder_log = encoder_default_log;

static void log(const char * fmt, ...) {
    if (!encoder_log) return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    encoder_log(buf);
}


static bool hann_window(int length, bool periodic, std::vector<float> & output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5*(1.0 - cosf((2.0*M_PI*i)/(length + offset)));
    }

    return true;
}


// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const std::vector<float> & in, std::vector<float> & out) {
    int N = in.size();

    out.resize(N*2);
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*cos_vals[idx]; // cos(t)
            im -= in[n]*sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(const std::vector<float> & in, std::vector<float> & out) {
    out.resize(in.size()*2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N%2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    even.reserve(N/2);
    odd.reserve(N/2);

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < N/2; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = cos_vals[idx]; // cos(t)
        float im = -sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + N/2) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + N/2) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}


// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
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
              encoder_mel & mel) {
    const int64_t t_start_us = ggml_time_us();

    // Hanning window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    std::vector<float> hann;
    hann_window(frame_size, true, hann);


    // Calculate the length of padding
    int64_t stage_1_pad = ENCODER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);


    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, std::cref(hann), samples_padded,
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    estate.t_mel_us += ggml_time_us() - t_start_us;

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}


// average the fabs of the signal
static std::vector<float> get_signal_energy(
                const float * signal, 
                int n_samples, 
                int n_samples_per_half_window) {
    const int hw = n_samples_per_half_window;

    std::vector<float> result(n_samples);

    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int j = -hw; j <= hw; j++) {
            if (i + j >= 0 && i + j < n_samples) {
                sum += fabs(signal[i + j]);
            }
        }
        result[i] = sum/(2*hw + 1);
    }

    return result;
}

int whisper_pcm_to_mel_with_state(
            struct encoder_context * ctx, 
            struct encoder_state * state, 
            const float * samples, 
            int n_samples, 
            int n_threads) {
    if (!log_mel_spectrogram(
                *state, samples, 
                n_samples, 
                ENCODER_SAMPLE_RATE, 
                ENCODER_N_FFT, 
                ENCODER_HOP_LENGTH, 
                ENCODER_N_MEL, 
                n_threads, 
                ctx->model.filters, 
                false, 
                state->mel)) {
        log("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}


static void log_mel_spectrogram_worker_thread(
                    int ith, 
                    const std::vector<float> & hann, 
                    const std::vector<float> & samples,
                    int n_samples, 
                    int frame_size, int frame_step, int n_threads,
                    const encoder_filters & filters, 
                    encoder_mel & mel) {
    std::vector<float> fft_in(frame_size, 0.0);
    std::vector<float> fft_out(2 * frame_step);
    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    int n_fft = 1 + (frame_size / 2);
    int i = ith;

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hanning window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in, fft_out);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < frame_size; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
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
              const int   n_threads){

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