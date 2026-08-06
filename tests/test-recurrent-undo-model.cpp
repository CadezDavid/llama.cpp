#include "common.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct error_stats {
    double sum_sq = 0.0;
    uint64_t count = 0;
    float max_abs = 0.0f;
    uint64_t nonfinite = 0;

    void add(const std::vector<float> & a, const std::vector<float> & b) {
        GGML_ASSERT(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            const float err = std::fabs(a[i] - b[i]);
            max_abs = std::max(max_abs, err);
            sum_sq += double(err)*err;
            nonfinite += !std::isfinite(a[i]);
        }
        count += a.size();
    }

    double rms() const { return count ? std::sqrt(sum_sq/count) : 0.0; }
};

static llama_memory_recurrent * get_recurrent(llama_context * ctx) {
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_recr();
    }
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

static bool truncate_attention_only(llama_context * ctx, llama_pos pos) {
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, pos, -1);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, pos, -1);
    }
    return true;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0) {
    if (tokens.empty()) {
        return true;
    }
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], pos0 + i, { 0 }, i + 1 == tokens.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_token argmax_token(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    GGML_ASSERT(logits);
    return std::max_element(logits, logits + n_vocab) - logits;
}

static error_stats compare_logits(llama_context * a, llama_context * b, int n_vocab) {
    const float * la = llama_get_logits_ith(a, -1);
    const float * lb = llama_get_logits_ith(b, -1);
    GGML_ASSERT(la && lb);
    std::vector<float> va(la, la + n_vocab);
    std::vector<float> vb(lb, lb + n_vocab);
    error_stats result;
    result.add(va, vb);
    return result;
}

static void compare_state(llama_memory_recurrent * a, llama_memory_recurrent * b,
        error_stats & r_error, error_stats & s_error) {
    for (size_t il = 0; il < a->s_l.size(); ++il) {
        if (!a->s_l[il]) {
            continue;
        }
        for (const auto & pair : { std::pair(a->r_l[il], b->r_l[il]),
                                  std::pair(a->s_l[il], b->s_l[il]) }) {
            const size_t row_bytes = ggml_row_size(pair.first->type, pair.first->ne[0]);
            std::vector<float> va(row_bytes/sizeof(float));
            std::vector<float> vb(row_bytes/sizeof(float));
            ggml_backend_tensor_get(pair.first, va.data(), 0, row_bytes);
            ggml_backend_tensor_get(pair.second, vb.data(), 0, row_bytes);
            if (pair.first == a->r_l[il]) {
                r_error.add(va, vb);
            } else {
                s_error.add(va, vb);
            }
        }
    }
}

static uint64_t hash_tokens(const std::vector<llama_token> & tokens) {
    uint64_t hash = 1469598103934665603ULL;
    for (llama_token token : tokens) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= (uint32_t(token) >> shift) & 0xff;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static void decay_diagnostics(llama_memory_recurrent * mem, uint32_t n_tokens,
        float & min_decay, double & max_log10_amplification) {
    min_decay = 1.0f;
    max_log10_amplification = 0.0;
    for (ggml_tensor * tensor : mem->undo_decay_l) {
        if (!tensor) {
            continue;
        }
        const int64_t n_heads = tensor->ne[1];
        std::vector<float> values(n_heads*n_tokens);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
        for (int64_t h = 0; h < n_heads; ++h) {
            double log10_amp = 0.0;
            for (uint32_t t = 0; t < n_tokens; ++t) {
                const float decay = values[t*n_heads + h];
                min_decay = std::min(min_decay, decay);
                log10_amp -= std::log10(std::max(decay, 1e-38f));
            }
            max_log10_amplification = std::max(max_log10_amplification, log10_amp);
        }
    }
}

static size_t compact_log_bytes(const llama_memory_recurrent * mem) {
    size_t result = 0;
    for (const auto & tensors : { &mem->undo_k_l, &mem->undo_delta_l, &mem->undo_decay_l, &mem->undo_conv_l }) {
        for (const ggml_tensor * tensor : *tensors) {
            if (tensor) {
                result += ggml_nbytes(tensor);
            }
        }
    }
    return result;
}

static bool save_state(llama_context * ctx, std::vector<uint8_t> & data) {
    data.resize(llama_state_get_size(ctx));
    return llama_state_get_data(ctx, data.data(), data.size()) == data.size();
}

static bool load_state(llama_context * ctx, const std::vector<uint8_t> & data) {
    return llama_state_set_data(ctx, data.data(), data.size()) == data.size();
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_gpu_layers = 999;
    int n_ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-c") || !strcmp(argv[i], "--ctx-size")) && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [-ngl N] [-c N]\n", argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-ngl N] [-c N]\n", argv[0]);
        return 1;
    }

    common_init();
    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        return 1;
    }
    if (!llama_model_is_hybrid(model.get())) {
        fprintf(stderr, "model must use hybrid attention and recurrent layers\n");
        return 1;
    }

    auto make_context = [&](uint32_t undo_capacity) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_seq_max = 1;
        cparams.n_rs_seq = 0;
        cparams.n_rs_undo = undo_capacity;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return llama_context_ptr(llama_init_from_model(model.get(), cparams));
    };

    // Use the same graph shape in both branches. The exact branch never calls
    // the inverse, but enabling its compact log prevents graph-shape effects
    // from being mistaken for inverse error.
    llama_context_ptr exact = make_context(16);
    llama_context_ptr inverse = make_context(16);
    if (!exact || !inverse || !get_recurrent(exact.get()) || !get_recurrent(inverse.get())) {
        fprintf(stderr, "failed to create compatible contexts\n");
        return 1;
    }

    const char * system = "You are a careful technical assistant. Give direct, practical explanations.";
    const char * user = "I am designing a small greenhouse for a windy coastal garden. Explain how to choose its orientation, foundation, and ventilation, including the tradeoffs.";
    const llama_chat_message messages[] = {
        { "system", system },
        { "user", user },
    };
    const char * tmpl = llama_model_chat_template(model.get(), nullptr);
    int32_t formatted_size = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (formatted_size < 0) {
        fprintf(stderr, "native chat template failed\n");
        return 1;
    }
    std::vector<char> formatted(formatted_size + 1);
    formatted_size = llama_chat_apply_template(tmpl, messages, 2, true, formatted.data(), formatted.size());
    const std::string prompt(formatted.data(), formatted_size);
    std::vector<llama_token> prompt_tokens = common_tokenize(exact.get(), prompt, true, true);
    if (prompt_tokens.empty() || !decode(exact.get(), prompt_tokens, 0) || !decode(inverse.get(), prompt_tokens, 0)) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    std::vector<uint8_t> saved_before;
    if (!save_state(exact.get(), saved_before)) {
        fprintf(stderr, "failed to save true pre-block state\n");
        return 1;
    }
    printf("prompt_tokens=%zu saved_context_bytes=%zu compact_log_bytes=%zu undo_capacity=16\n",
            prompt_tokens.size(), saved_before.size(), compact_log_bytes(get_recurrent(inverse.get())));

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::vector<llama_token> block;
    for (int i = 0; i < 16; ++i) {
        const llama_token token = argmax_token(exact.get(), n_vocab);
        block.push_back(token);
        if (!decode(exact.get(), { token }, prompt_tokens.size() + i)) {
            fprintf(stderr, "failed to generate calibration block\n");
            return 1;
        }
    }

    bool passed = true;
    for (uint32_t depth : { 1u, 4u, 8u, 16u }) {
        if (!load_state(exact.get(), saved_before) || !load_state(inverse.get(), saved_before)) {
            fprintf(stderr, "state restore failed at depth %u\n", depth);
            return 1;
        }
        const uint32_t prefix_count = 16 - depth;
        const std::vector<llama_token> prefix(block.begin(), block.begin() + prefix_count);
        const std::vector<llama_token> tail(block.begin() + prefix_count, block.end());
        if (!decode(inverse.get(), prefix, prompt_tokens.size()) ||
                !decode(exact.get(), prefix, prompt_tokens.size()) ||
                !decode(inverse.get(), tail, prompt_tokens.size() + prefix_count)) {
            fprintf(stderr, "block decode failed at depth %u\n", depth);
            return 1;
        }

        llama_memory_recurrent * inverse_recr = get_recurrent(inverse.get());
        llama_synchronize(inverse.get());
        float min_decay = 1.0f;
        double max_log10_amp = 0.0;
        decay_diagnostics(inverse_recr, depth, min_decay, max_log10_amp);
        float gdn_ms = 0.0f;
        float conv_ms = 0.0f;
        const int64_t inverse_start_us = ggml_time_us();
        if (!inverse_recr->undo(depth, &gdn_ms, &conv_ms)) {
            fprintf(stderr, "inverse kernels failed at depth %u\n", depth);
            return 1;
        }
        const double inverse_wall_ms = (ggml_time_us() - inverse_start_us)/1000.0;
        const llama_pos next_pos = prompt_tokens.size() + prefix_count;
        if (!truncate_attention_only(inverse.get(), next_pos) ||
                !inverse_recr->set_pos_after_undo(0, next_pos - 1)) {
            fprintf(stderr, "cache truncation failed at depth %u\n", depth);
            return 1;
        }

        llama_synchronize(exact.get());
        llama_synchronize(inverse.get());
        error_stats r_error;
        error_stats s_error;
        compare_state(inverse_recr, get_recurrent(exact.get()), r_error, s_error);

        // The inverse branch's old logits correspond to the discarded block.
        // Feed both branches the exact next token once, then compare every
        // subsequent distribution and greedy choice from identical prefixes.
        const llama_token bridge = argmax_token(exact.get(), n_vocab);
        if (!decode(exact.get(), { bridge }, next_pos) || !decode(inverse.get(), { bridge }, next_pos)) {
            fprintf(stderr, "bridge decode failed at depth %u\n", depth);
            return 1;
        }

        std::vector<llama_token> exact_out = { bridge };
        std::vector<llama_token> inverse_out = { bridge };
        error_stats logits_error;
        error_stats first_logits_error;
        uint32_t token_mismatches = 0;
        int first_mismatch_step = -1;
        llama_token first_exact_token = -1;
        llama_token first_inverse_token = -1;
        for (int step = 0; step < 16; ++step) {
            const error_stats step_error = compare_logits(inverse.get(), exact.get(), n_vocab);
            if (step == 0) {
                first_logits_error = step_error;
            }
            logits_error.sum_sq += step_error.sum_sq;
            logits_error.count += step_error.count;
            logits_error.max_abs = std::max(logits_error.max_abs, step_error.max_abs);
            logits_error.nonfinite += step_error.nonfinite;

            const llama_token exact_token = argmax_token(exact.get(), n_vocab);
            const llama_token inverse_token = argmax_token(inverse.get(), n_vocab);
            if (step == 0) {
                first_exact_token = exact_token;
                first_inverse_token = inverse_token;
            }
            if (exact_token != inverse_token && first_mismatch_step < 0) {
                first_mismatch_step = step;
            }
            token_mismatches += exact_token != inverse_token;
            exact_out.push_back(exact_token);
            inverse_out.push_back(inverse_token);
            if (!decode(exact.get(), { exact_token }, next_pos + 1 + step) ||
                    !decode(inverse.get(), { inverse_token }, next_pos + 1 + step)) {
                fprintf(stderr, "continuation decode failed at depth %u step %d\n", depth, step);
                return 1;
            }
        }

        std::vector<llama_token> exact_complete(prompt_tokens);
        std::vector<llama_token> inverse_complete(prompt_tokens);
        exact_complete.insert(exact_complete.end(), block.begin(), block.begin() + prefix_count);
        inverse_complete.insert(inverse_complete.end(), block.begin(), block.begin() + prefix_count);
        exact_complete.insert(exact_complete.end(), exact_out.begin(), exact_out.end());
        inverse_complete.insert(inverse_complete.end(), inverse_out.begin(), inverse_out.end());
        const uint64_t exact_hash = hash_tokens(exact_complete);
        const uint64_t inverse_hash = hash_tokens(inverse_complete);
        printf("depth=%2u inverse_wall_ms=%.4f kernel_ms=%.4f gdn_ms=%.4f conv_ms=%.4f "
               "R_max=%.9g R_rms=%.9g S_max=%.9g S_rms=%.9g "
               "min_decay=%.9g max_log10_amp=%.3f "
               "first_logits_max=%.9g first_logits_rms=%.9g first_tokens=%d/%d "
               "logits_max=%.9g logits_rms=%.9g token_mismatches=%u first_mismatch=%d "
               "exact_hash=%016llx inverse_hash=%016llx\n",
                depth, inverse_wall_ms, gdn_ms + conv_ms, gdn_ms, conv_ms,
                r_error.max_abs, r_error.rms(), s_error.max_abs, s_error.rms(),
                min_decay, max_log10_amp,
                first_logits_error.max_abs, first_logits_error.rms(), first_exact_token, first_inverse_token,
                logits_error.max_abs, logits_error.rms(), token_mismatches, first_mismatch_step,
                (unsigned long long) exact_hash, (unsigned long long) inverse_hash);
        passed &= r_error.nonfinite == 0 && s_error.nonfinite == 0 && logits_error.nonfinite == 0 &&
                token_mismatches == 0 && exact_hash == inverse_hash;
    }

    return passed ? 0 : 1;
}
