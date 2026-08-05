#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct eval_options {
    float sparsity = 0.0f;
    int32_t block_size = 1;
    float proxy_input_sparsity = 0.0f;
    int32_t proxy_block_size = 1;
    bool proxy_use_values = false;
    int32_t prefix_tokens = 0;
    int32_t eval_tokens = 256;
};

static bool parse_i32(const char * text, int32_t & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) {
        return false;
    }
    value = (int32_t) parsed;
    return true;
}

static bool parse_f32(const char * text, float & value) {
    char * end = nullptr;
    const float parsed = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

static bool parse_options(
        int argc,
        char ** argv,
        eval_options & options,
        std::vector<char *> & filtered) {
    filtered = { argv[0] };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto get_value = [&](const char * name) -> const char * {
            const std::string prefix = std::string(name) + "=";
            if (arg.compare(0, prefix.size(), prefix) == 0) {
                return argv[i] + prefix.size();
            }
            if (arg == name && i + 1 < argc) {
                return argv[++i];
            }
            return nullptr;
        };

        if (const char * value = get_value("--ffn-sparsity")) {
            if (!parse_f32(value, options.sparsity)) {
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--ffn-block-size")) {
            if (!parse_i32(value, options.block_size)) {
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--ffn-proxy-input-sparsity")) {
            if (!parse_f32(value, options.proxy_input_sparsity)) {
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--ffn-proxy-block-size")) {
            if (!parse_i32(value, options.proxy_block_size)) {
                return false;
            }
            continue;
        }
        if (arg == "--ffn-proxy-use-values") {
            options.proxy_use_values = true;
            continue;
        }
        if (const char * value = get_value("--prefix-tokens")) {
            if (!parse_i32(value, options.prefix_tokens)) {
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--eval-tokens")) {
            if (!parse_i32(value, options.eval_tokens)) {
                return false;
            }
            continue;
        }

        filtered.push_back(argv[i]);
    }

    return options.sparsity >= 0.0f && options.sparsity < 1.0f &&
            options.block_size >= 1 && options.proxy_input_sparsity >= 0.0f &&
            options.proxy_input_sparsity < 1.0f && options.proxy_block_size >= 1 &&
            options.prefix_tokens >= 0 && options.eval_tokens >= 1;
}

static bool decode_range(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int32_t first,
        int32_t count) {
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (int32_t offset = 0; offset < count; offset += n_batch) {
        const int32_t n_tokens = std::min(n_batch, count - offset);
        common_batch_clear(batch);
        for (int32_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch, tokens[first + offset + i], first + offset + i, { 0 }, false);
        }
        batch.logits[n_tokens - 1] = true;
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);
    return true;
}

static bool decode_one(llama_context * ctx, llama_token token, int32_t pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return ok;
}

static double log_sum_exp(const float * logits, int32_t n_vocab) {
    const float max_logit = *std::max_element(logits, logits + n_vocab);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - max_logit);
    }
    return max_logit + std::log(sum);
}

int main(int argc, char ** argv) {
    eval_options options;
    std::vector<char *> filtered;
    if (!parse_options(argc, argv, options, filtered)) {
        LOG_ERR("invalid FFN sparse evaluation options\n");
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.n_parallel = 1;
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    if (!init) {
        return 1;
    }

    llama_model * model = init->model();
    llama_context * dense = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    auto sparse_params = common_context_params_to_llama(params);
    llama_context_ptr sparse(llama_init_from_model(model, sparse_params));
    if (!sparse) {
        LOG_ERR("failed to create sparse comparison context\n");
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(dense, params.prompt, true, true);
    if (options.prefix_tokens == 0) {
        options.prefix_tokens = (int32_t) tokens.size() - options.eval_tokens - 1;
    }
    if (options.prefix_tokens < 1 ||
            options.prefix_tokens + options.eval_tokens + 1 > (int32_t) tokens.size() ||
            options.prefix_tokens + options.eval_tokens > (int32_t) llama_n_ctx(dense)) {
        LOG_ERR("prompt does not contain the requested prefix and evaluation window\n");
        return 1;
    }

    if (!decode_range(dense, tokens, 0, options.prefix_tokens) ||
            !decode_range(sparse.get(), tokens, 0, options.prefix_tokens)) {
        LOG_ERR("failed to decode dense prefix\n");
        return 1;
    }

    if (!llama_vegas_set_ffn_oracle_sparsity(sparse.get(), options.sparsity) ||
            !llama_vegas_set_ffn_oracle_block_size(sparse.get(), options.block_size) ||
            !llama_vegas_set_ffn_proxy(
                    sparse.get(), options.proxy_input_sparsity, options.proxy_block_size,
                    options.proxy_use_values)) {
        LOG_ERR("failed to configure sparse comparison context\n");
        return 1;
    }

    double dense_nll = 0.0;
    double sparse_nll = 0.0;
    double kl_sum = 0.0;
    double max_kl = 0.0;
    double l1_prob_sum = 0.0;
    int32_t top1_equal = 0;
    int32_t dense_true = 0;
    int32_t sparse_true = 0;

    for (int32_t i = 0; i < options.eval_tokens; ++i) {
        const int32_t pos = options.prefix_tokens + i;
        if (!decode_one(dense, tokens[pos], pos) || !decode_one(sparse.get(), tokens[pos], pos)) {
            LOG_ERR("failed to decode evaluation token %d\n", i);
            return 1;
        }

        const float * dense_logits = llama_get_logits_ith(dense, 0);
        const float * sparse_logits = llama_get_logits_ith(sparse.get(), 0);
        const double dense_lse = log_sum_exp(dense_logits, n_vocab);
        const double sparse_lse = log_sum_exp(sparse_logits, n_vocab);
        const llama_token target = tokens[pos + 1];

        dense_nll += dense_lse - dense_logits[target];
        sparse_nll += sparse_lse - sparse_logits[target];

        const int32_t dense_top = (int32_t) (std::max_element(
                dense_logits, dense_logits + n_vocab) - dense_logits);
        const int32_t sparse_top = (int32_t) (std::max_element(
                sparse_logits, sparse_logits + n_vocab) - sparse_logits);
        top1_equal += dense_top == sparse_top;
        dense_true += dense_top == target;
        sparse_true += sparse_top == target;

        double kl = 0.0;
        double l1 = 0.0;
        for (int32_t token = 0; token < n_vocab; ++token) {
            const double p = std::exp(dense_logits[token] - dense_lse);
            const double q = std::exp(sparse_logits[token] - sparse_lse);
            if (p > 0.0) {
                kl += p * ((dense_logits[token] - dense_lse) -
                        (sparse_logits[token] - sparse_lse));
            }
            l1 += std::abs(p - q);
        }
        kl_sum += kl;
        max_kl = std::max(max_kl, kl);
        l1_prob_sum += l1;
    }

    const double n = options.eval_tokens;
    std::printf(
            "FFN_SPARSE_EVAL {\"sparsity\":%.6f,\"block_size\":%d,"
            "\"proxy_input_sparsity\":%.6f,\"proxy_block_size\":%d,\"proxy_use_values\":%s,"
            "\"prefix_tokens\":%d,\"eval_tokens\":%d,"
            "\"dense_ppl\":%.8f,\"sparse_ppl\":%.8f,\"ppl_ratio\":%.8f,"
            "\"mean_kl\":%.10f,\"max_kl\":%.10f,\"mean_l1\":%.10f,"
            "\"top1_agreement\":%.8f,\"dense_teacher_top1\":%.8f,"
            "\"sparse_teacher_top1\":%.8f}\n",
            options.sparsity, options.block_size,
            options.proxy_input_sparsity, options.proxy_block_size,
            options.proxy_use_values ? "true" : "false",
            options.prefix_tokens, options.eval_tokens,
            std::exp(dense_nll / n), std::exp(sparse_nll / n),
            std::exp((sparse_nll - dense_nll) / n), kl_sum / n, max_kl, l1_prob_sum / n,
            top1_equal / n, dense_true / n, sparse_true / n);

    return 0;
}
