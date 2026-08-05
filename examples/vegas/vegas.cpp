#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "adaptive-gamma.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

enum class vegas_run_mode {
    baseline,
    dense_spec,
    vegas,
    mtp,
    mtp_vegas,
    mtp_auto,
};

struct vegas_options {
    vegas_run_mode mode = vegas_run_mode::vegas;
    float sparse_ratio = 0.07f;
    int32_t min_tokens = 256;
    int32_t max_tokens = 0;
    int32_t gamma = 8;
    int32_t selection_layer = -1;
    int32_t anchor_tokens = 0;
    int32_t refresh_interval = 1;
    int32_t mtp_ubatch = 128;
    int32_t prompt_tokens = 0;
    bool auto_policy = false;
    bool adaptive_gamma = false;
    float adaptive_beta = 0.9f;
    bool quiet = false;
};

struct token_distribution {
    llama_token sampled = LLAMA_TOKEN_NULL;
    std::vector<std::pair<llama_token, float>> probs;

    float probability(llama_token id) const {
        for (const auto & entry : probs) {
            if (entry.first == id) {
                return entry.second;
            }
        }
        return 0.0f;
    }
};

struct vegas_metrics {
    int64_t prompt_us = 0;
    int64_t initial_select_us = 0;
    int64_t draft_us = 0;
    int64_t verify_us = 0;
    int64_t collect_us = 0;
    int64_t sample_us = 0;
    int64_t rollback_us = 0;
    int64_t total_us = 0;

    int32_t n_prompt = 0;
    int32_t n_predict = 0;
    int32_t n_cycles = 0;
    int32_t n_drafted = 0;
    int32_t n_accepted = 0;
    int32_t n_rejected = 0;
    int32_t n_reused = 0;
    uint64_t output_hash = UINT64_C(1469598103934665603);
    std::string adaptive_summary;
};

static const char * mode_name(vegas_run_mode mode) {
    switch (mode) {
        case vegas_run_mode::baseline:   return "baseline";
        case vegas_run_mode::dense_spec: return "dense-spec";
        case vegas_run_mode::vegas:      return "vegas";
        case vegas_run_mode::mtp:        return "mtp";
        case vegas_run_mode::mtp_vegas:  return "mtp-vegas";
        case vegas_run_mode::mtp_auto:   return "mtp-auto";
    }
    return "unknown";
}

static bool mode_uses_vegas(vegas_run_mode mode) {
    return mode == vegas_run_mode::vegas || mode == vegas_run_mode::mtp_vegas ||
            mode == vegas_run_mode::mtp_auto;
}

static bool mode_uses_mtp(vegas_run_mode mode) {
    return mode == vegas_run_mode::mtp || mode == vegas_run_mode::mtp_vegas ||
            mode == vegas_run_mode::mtp_auto;
}

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

static bool parse_vegas_options(
        int argc,
        char ** argv,
        vegas_options & options,
        std::vector<char *> & filtered) {
    filtered.clear();
    filtered.push_back(argv[0]);

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

        if (arg == "--vegas-quiet") {
            options.quiet = true;
            continue;
        }
        if (arg == "--vegas-adaptive-gamma") {
            options.adaptive_gamma = true;
            continue;
        }

        if (const char * value = get_value("--vegas-mode")) {
            if (std::strcmp(value, "baseline") == 0) {
                options.mode = vegas_run_mode::baseline;
            } else if (std::strcmp(value, "dense-spec") == 0) {
                options.mode = vegas_run_mode::dense_spec;
            } else if (std::strcmp(value, "vegas") == 0) {
                options.mode = vegas_run_mode::vegas;
            } else if (std::strcmp(value, "mtp") == 0) {
                options.mode = vegas_run_mode::mtp;
            } else if (std::strcmp(value, "mtp-vegas") == 0) {
                options.mode = vegas_run_mode::mtp_vegas;
            } else if (std::strcmp(value, "mtp-auto") == 0) {
                options.mode = vegas_run_mode::mtp_auto;
                options.auto_policy = true;
            } else {
                LOG_ERR("invalid --vegas-mode: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-ratio")) {
            if (!parse_f32(value, options.sparse_ratio)) {
                LOG_ERR("invalid --vegas-ratio: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-min-tokens")) {
            if (!parse_i32(value, options.min_tokens)) {
                LOG_ERR("invalid --vegas-min-tokens: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-max-tokens")) {
            if (!parse_i32(value, options.max_tokens)) {
                LOG_ERR("invalid --vegas-max-tokens: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-gamma")) {
            if (!parse_i32(value, options.gamma)) {
                LOG_ERR("invalid --vegas-gamma: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-adaptive-beta")) {
            if (!parse_f32(value, options.adaptive_beta)) {
                LOG_ERR("invalid --vegas-adaptive-beta: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-selection-layer")) {
            if (!parse_i32(value, options.selection_layer)) {
                LOG_ERR("invalid --vegas-selection-layer: %s\n", value);
                return false;
            }
            continue;
        }

        if (const char * value = get_value("--vegas-anchor-tokens")) {
            if (!parse_i32(value, options.anchor_tokens)) {
                LOG_ERR("invalid --vegas-anchor-tokens: %s\n", value);
                return false;
            }
            continue;
        }

        if (const char * value = get_value("--vegas-refresh-interval")) {
            if (!parse_i32(value, options.refresh_interval)) {
                LOG_ERR("invalid --vegas-refresh-interval: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-prompt-tokens")) {
            if (!parse_i32(value, options.prompt_tokens)) {
                LOG_ERR("invalid --vegas-prompt-tokens: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-mtp-ubatch")) {
            if (!parse_i32(value, options.mtp_ubatch)) {
                LOG_ERR("invalid --vegas-mtp-ubatch: %s\n", value);
                return false;
            }
            continue;
        }

        filtered.push_back(argv[i]);
    }

    if (!(options.sparse_ratio > 0.0f && options.sparse_ratio <= 1.0f) ||
            options.min_tokens < 1 || options.max_tokens < 0 || options.gamma < 1 ||
            options.anchor_tokens < 0 || options.refresh_interval < 1 ||
            options.mtp_ubatch < 1 || options.prompt_tokens < 0 || options.prompt_tokens == 1 ||
            !(options.adaptive_beta >= 0.0f && options.adaptive_beta < 1.0f)) {
        LOG_ERR("invalid Vegas configuration\n");
        return false;
    }

    return true;
}

static token_distribution sample_distribution(
        common_sampler * sampler,
        llama_context * ctx,
        int32_t idx,
        bool with_probs) {
    token_distribution result;
    result.sampled = common_sampler_sample(sampler, ctx, idx, true);

    if (!with_probs) {
        return result;
    }

    const auto * candidates = common_sampler_get_candidates(sampler, false);
    result.probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].p > 0.0f) {
            result.probs.emplace_back(candidates->data[i].id, candidates->data[i].p);
        }
    }

    return result;
}

static llama_token sample_residual(
        const token_distribution & target,
        const token_distribution & draft,
        int32_t n_vocab,
        std::mt19937 & rng) {
    std::vector<double> residual(n_vocab, 0.0);
    for (const auto & entry : target.probs) {
        residual[entry.first] = entry.second;
    }
    for (const auto & entry : draft.probs) {
        residual[entry.first] = std::max(0.0, residual[entry.first] - entry.second);
    }

    double sum = 0.0;
    for (double value : residual) {
        sum += value;
    }
    GGML_ASSERT(sum > 0.0);

    std::discrete_distribution<int32_t> distribution(residual.begin(), residual.end());
    return (llama_token) distribution(rng);
}

static bool decode_one(llama_context * ctx, llama_batch & batch, llama_token token, int32_t pos) {
    common_batch_clear(batch);
    common_batch_add(batch, token, pos, { 0 }, true);
    return llama_decode(ctx, batch) == 0;
}

static void resolve_auto_policy(
        vegas_options & options,
        common_params & params,
        const llama_model * model,
        int32_t n_prompt) {
    if (options.mode != vegas_run_mode::mtp_auto) {
        return;
    }

    const bool separate_draft = params.speculative.has_dft();
    const bool moe = llama_model_n_expert(model) > 0;
    bool sparse = false;

    if (separate_draft) {
        sparse = n_prompt >= 96 * 1024;
        options.gamma = 1;
        options.sparse_ratio = 0.03f;
        options.selection_layer = llama_model_n_layer(model) - 1;
        options.refresh_interval = 1;
    } else if (moe) {
        sparse = n_prompt >= 48 * 1024 &&
                params.cache_type_k == GGML_TYPE_Q4_0 && params.cache_type_v == GGML_TYPE_Q4_0;
        options.gamma = sparse ? 5 : 2;
        if (params.cache_type_k == GGML_TYPE_Q8_0 && params.cache_type_v == GGML_TYPE_TURBO4_0) {
            params.speculative.draft.cache_type_k = GGML_TYPE_Q4_0;
            params.speculative.draft.cache_type_v = GGML_TYPE_Q4_0;
        }
        options.sparse_ratio = 0.03f;
        options.selection_layer = llama_model_n_layer(model) - 1;
        options.refresh_interval = 1;
    } else {
        const bool q4_cache =
                params.cache_type_k == GGML_TYPE_Q4_0 && params.cache_type_v == GGML_TYPE_Q4_0;
        sparse = n_prompt >= (q4_cache ? 48 : 96) * 1024;
        options.gamma = 4;
        options.sparse_ratio = std::clamp(0.04f + n_prompt / 2000000.0f, 0.05f, 0.10f);
        options.selection_layer = std::max(0, (llama_model_n_layer(model) - 1) / 4);
        options.refresh_interval = 2;
    }

    options.mode = sparse ? vegas_run_mode::mtp_vegas : vegas_run_mode::mtp;
    if (!sparse) {
        options.selection_layer = -1;
    }
    LOG_INF("Vegas auto: mode=%s gamma<=%d ratio=%.3f selection-layer=%d refresh=%d draft-cache=%s/%s\n",
            mode_name(options.mode), options.gamma, options.sparse_ratio,
            options.selection_layer, options.refresh_interval,
            ggml_type_name(params.speculative.draft.cache_type_k),
            ggml_type_name(params.speculative.draft.cache_type_v));
}

static bool decode_prompt(
        llama_context * ctx,
        std::vector<llama_token> & tokens,
        int32_t n_tokens,
        common_speculative * spec = nullptr) {
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int32_t offset = 0; offset < n_tokens; offset += n_batch) {
        const int32_t count = std::min(n_batch, n_tokens - offset);
        common_batch_clear(batch);
        for (int32_t i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[offset + i], offset + i, { 0 }, true);
        }
        if (llama_decode(ctx, batch) != 0 || !common_speculative_process(spec, batch)) {
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);
    return true;
}

static bool remove_after(llama_context * ctx, int32_t pos) {
    return llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1);
}

static void record_token(
        const llama_context * ctx,
        llama_token token,
        bool quiet,
        vegas_metrics & metrics) {
    metrics.output_hash ^= (uint32_t) token;
    metrics.output_hash *= UINT64_C(1099511628211);
    if (!quiet) {
        LOG("%s", common_token_to_piece(ctx, token).c_str());
    }
}

static bool run_baseline(
        llama_context * ctx,
        const llama_vocab * vocab,
        common_sampler * sampler,
        llama_token id_last,
        int32_t n_past,
        const common_params & params,
        const vegas_options & options,
        llama_batch & batch,
        vegas_metrics & metrics) {
    const int64_t start = ggml_time_us();
    bool has_eog = false;

    while (!has_eog && (params.n_predict < 0 || metrics.n_predict < params.n_predict)) {
        const int64_t decode_start = ggml_time_us();
        if (!decode_one(ctx, batch, id_last, n_past)) {
            return false;
        }
        llama_synchronize(ctx);
        metrics.verify_us += ggml_time_us() - decode_start;

        const int64_t sample_start = ggml_time_us();
        const llama_token next = common_sampler_sample(sampler, ctx, 0, true);
        common_sampler_accept(sampler, next, true);
        metrics.sample_us += ggml_time_us() - sample_start;

        ++n_past;
        ++metrics.n_predict;
        id_last = next;
        record_token(ctx, id_last, options.quiet, metrics);
        has_eog = llama_vocab_is_eog(vocab, id_last);
    }

    metrics.total_us = ggml_time_us() - start;
    return true;
}

static bool run_speculative(
        llama_context * ctx,
        const llama_vocab * vocab,
        common_sampler * sampler,
        llama_token id_last,
        int32_t n_past,
        const common_params & params,
        const vegas_options & options,
        llama_batch & batch,
        vegas_metrics & metrics) {
    const bool sparse = options.mode == vegas_run_mode::vegas;
    const bool stochastic = params.sampling.temp > 0.0f;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t seed = params.sampling.seed == LLAMA_DEFAULT_SEED ?
        std::random_device{}() : params.sampling.seed;
    std::mt19937 rng(seed ^ 0x9e3779b9U);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    int32_t selected_prefix = n_past;
    bool has_eog = false;
    const int64_t start = ggml_time_us();

    while (!has_eog && (params.n_predict < 0 || metrics.n_predict < params.n_predict)) {
        const int32_t remaining = params.n_predict < 0 ? INT32_MAX : params.n_predict - metrics.n_predict;
        if (remaining == 1) {
            if (sparse) {
                llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_DISABLED, 0);
            }
            const int64_t decode_start = ggml_time_us();
            if (!decode_one(ctx, batch, id_last, n_past)) {
                return false;
            }
            llama_synchronize(ctx);
            metrics.verify_us += ggml_time_us() - decode_start;

            const int64_t sample_start = ggml_time_us();
            id_last = common_sampler_sample(sampler, ctx, 0, true);
            common_sampler_accept(sampler, id_last, true);
            metrics.sample_us += ggml_time_us() - sample_start;
            ++n_past;
            ++metrics.n_predict;
            record_token(ctx, id_last, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, id_last);
            break;
        }

        const int32_t gamma = std::min(options.gamma, remaining - 1);
        common_sampler_ptr draft_sampler(common_sampler_clone(sampler));
        std::vector<llama_token> draft;
        std::vector<token_distribution> draft_probs;
        draft.reserve(gamma);
        draft_probs.reserve(gamma);

        const int64_t draft_start = ggml_time_us();
        if (sparse) {
            llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_DRAFT, selected_prefix);
        } else {
            llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_DISABLED, 0);
        }

        llama_token input = id_last;
        for (int32_t i = 0; i < gamma; ++i) {
            if (!decode_one(ctx, batch, input, n_past + i)) {
                return false;
            }

            auto distribution = sample_distribution(draft_sampler.get(), ctx, 0, stochastic);
            input = distribution.sampled;
            common_sampler_accept(draft_sampler.get(), input, true);
            draft.push_back(input);
            draft_probs.push_back(std::move(distribution));
        }
        metrics.draft_us += ggml_time_us() - draft_start;
        metrics.n_drafted += gamma;

        const int64_t rollback_start = ggml_time_us();
        if (!remove_after(ctx, n_past)) {
            LOG_ERR("failed to roll back sparse draft state\n");
            return false;
        }
        metrics.rollback_us += ggml_time_us() - rollback_start;

        if (sparse) {
            llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_VERIFY, n_past + 1);
        }
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past, { 0 }, true);
        for (int32_t i = 0; i < gamma; ++i) {
            common_batch_add(batch, draft[i], n_past + i + 1, { 0 }, true);
        }

        const int64_t verify_start = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        llama_synchronize(ctx);
        metrics.verify_us += ggml_time_us() - verify_start;

        if (sparse) {
            const int64_t collect_start = ggml_time_us();
            if (!llama_vegas_collect_indices(ctx)) {
                LOG_ERR("failed to collect Vegas indices\n");
                return false;
            }
            metrics.collect_us += ggml_time_us() - collect_start;
            selected_prefix = n_past + 1;
        }

        int32_t accepted = 0;
        llama_token next = LLAMA_TOKEN_NULL;
        const int64_t sample_start = ggml_time_us();
        for (int32_t i = 0; i < gamma; ++i) {
            const auto target = sample_distribution(sampler, ctx, i, stochastic);
            bool accept = false;

            if (stochastic) {
                const float p = target.probability(draft[i]);
                const float q = draft_probs[i].probability(draft[i]);
                GGML_ASSERT(q > 0.0f);
                accept = uniform(rng) <= std::min(1.0f, p / q);
            } else {
                accept = target.sampled == draft[i];
            }

            if (accept) {
                next = draft[i];
                common_sampler_accept(sampler, next, true);
                ++accepted;
                ++metrics.n_accepted;
                ++metrics.n_predict;
                record_token(ctx, next, options.quiet, metrics);
                if (llama_vocab_is_eog(vocab, next)) {
                    has_eog = true;
                    break;
                }
                continue;
            }

            next = stochastic ? sample_residual(target, draft_probs[i], n_vocab, rng) : target.sampled;
            common_sampler_accept(sampler, next, true);
            ++metrics.n_rejected;
            ++metrics.n_predict;
            record_token(ctx, next, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, next);
            break;
        }

        if (!has_eog && accepted == gamma) {
            next = common_sampler_sample(sampler, ctx, gamma, true);
            common_sampler_accept(sampler, next, true);
            ++metrics.n_predict;
            record_token(ctx, next, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, next);
        }
        metrics.sample_us += ggml_time_us() - sample_start;

        n_past += accepted + 1;
        id_last = next;
        ++metrics.n_cycles;

        const int64_t rollback_verify_start = ggml_time_us();
        if (!remove_after(ctx, n_past)) {
            LOG_ERR("failed to roll back rejected verification state\n");
            return false;
        }
        metrics.rollback_us += ggml_time_us() - rollback_verify_start;
    }

    metrics.total_us = ggml_time_us() - start;
    return true;
}

static bool run_mtp(
        llama_context * ctx,
        llama_context * ctx_dft,
        common_speculative * spec,
        const llama_vocab * vocab,
        common_sampler * sampler,
        llama_token id_last,
        int32_t n_past,
        const common_params & params,
        const vegas_options & options,
        llama_batch & batch,
        llama_tokens & history,
        vegas_metrics & metrics) {
    const bool sparse = options.mode == vegas_run_mode::mtp_vegas;
    bool has_eog = false;
    const int32_t draft_capacity = options.adaptive_gamma ? vegas_adaptive_gamma::max_gamma : options.gamma;
    vegas_adaptive_gamma adaptive(options.adaptive_beta, options.gamma);
    const int64_t start = ggml_time_us();

    while (!has_eog && (params.n_predict < 0 || metrics.n_predict < params.n_predict)) {
        const int32_t remaining = params.n_predict < 0 ? INT32_MAX : params.n_predict - metrics.n_predict;
        bool cycle_sparse = sparse;
        int32_t cycle_gamma = draft_capacity;
        if (options.adaptive_gamma && remaining > 1) {
            adaptive.begin_cycle();
            cycle_sparse = sparse && adaptive.planned_sparse();
            cycle_gamma = adaptive.planned_gamma();
        }
        const int32_t n_max = remaining > 1 ? std::min(cycle_gamma, remaining - 1) : 0;
        llama_tokens draft;
        int64_t cycle_draft_us = 0;
        int64_t cycle_verify_us = 0;
        int64_t cycle_collect_us = 0;

        if (n_max > 0) {
            if (cycle_sparse && metrics.n_cycles % options.refresh_interval == 0) {
                if (!llama_vegas_copy_indices(ctx_dft, ctx)) {
                    LOG_ERR("failed to copy Vegas indices to MTP context\n");
                    return false;
                }
            } else if (cycle_sparse && !llama_vegas_resume_draft(ctx_dft)) {
                LOG_ERR("failed to resume Vegas indices in MTP context\n");
                return false;
            } else if (!cycle_sparse) {
                llama_vegas_pause(ctx_dft);
            }

            auto & draft_params = common_speculative_get_draft_params(spec, 0);
            draft_params = {
                /* .drafting = */ true,
                /* .n_max    = */ n_max,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &history,
                /* .result   = */ &draft,
            };
            if (options.adaptive_gamma) {
                draft_params.observer = vegas_adaptive_gamma::observer;
                draft_params.observer_userdata = &adaptive;
            }

            const int64_t draft_start = ggml_time_us();
            common_speculative_draft(spec);
            llama_synchronize(ctx_dft);
            cycle_draft_us = ggml_time_us() - draft_start;
            metrics.draft_us += cycle_draft_us;
            metrics.n_drafted += (int32_t) draft.size();

            llama_vegas_pause(ctx_dft);
            const int64_t rollback_start = ggml_time_us();
            if (!remove_after(ctx_dft, n_past)) {
                LOG_ERR("failed to roll back MTP draft state\n");
                return false;
            }
            metrics.rollback_us += ggml_time_us() - rollback_start;
        }

        const bool refresh_indices = cycle_sparse &&
                (metrics.n_cycles + 1) % options.refresh_interval == 0;
        if (refresh_indices) {
            llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_VERIFY, n_past + 1);
        } else if (sparse) {
            // Keep the last verified prefix and plan available for a later
            // sparse cycle while running this verification densely.
            llama_vegas_pause(ctx);
        }

        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], n_past + (int32_t) i + 1, { 0 }, true);
        }

        const int64_t verify_start = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        llama_synchronize(ctx);
        cycle_verify_us = ggml_time_us() - verify_start;
        metrics.verify_us += cycle_verify_us;

        if (refresh_indices) {
            const int64_t collect_start = ggml_time_us();
            if (!llama_vegas_collect_indices(ctx)) {
                LOG_ERR("failed to collect Vegas indices\n");
                return false;
            }
            cycle_collect_us = ggml_time_us() - collect_start;
            metrics.collect_us += cycle_collect_us;
        }

        if (!common_speculative_process(spec, batch)) {
            LOG_ERR("failed to process MTP verification batch\n");
            return false;
        }

        int32_t accepted = 0;
        llama_token next = LLAMA_TOKEN_NULL;
        const int64_t sample_start = ggml_time_us();
        for (size_t i = 0; i < draft.size(); ++i) {
            const llama_token target = common_sampler_sample(sampler, ctx, (int32_t) i, true);
            if (target != draft[i]) {
                next = target;
                common_sampler_accept(sampler, next, true);
                ++metrics.n_rejected;
                ++metrics.n_predict;
                history.push_back(next);
                record_token(ctx, next, options.quiet, metrics);
                has_eog = llama_vocab_is_eog(vocab, next);
                break;
            }

            next = draft[i];
            common_sampler_accept(sampler, next, true);
            ++accepted;
            ++metrics.n_accepted;
            ++metrics.n_predict;
            history.push_back(next);
            record_token(ctx, next, options.quiet, metrics);
            if (llama_vocab_is_eog(vocab, next)) {
                has_eog = true;
                break;
            }
        }

        if (!has_eog && accepted == (int32_t) draft.size()) {
            next = common_sampler_sample(sampler, ctx, (int32_t) draft.size(), true);
            common_sampler_accept(sampler, next, true);
            ++metrics.n_predict;
            history.push_back(next);
            record_token(ctx, next, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, next);
        }
        metrics.sample_us += ggml_time_us() - sample_start;

        common_speculative_accept(spec, 0, accepted);

        if (options.adaptive_gamma && !draft.empty()) {
            adaptive.finish_cycle(
                    (int32_t) draft.size(), accepted, cycle_draft_us,
                    cycle_verify_us + cycle_collect_us, cycle_sparse);
            metrics.adaptive_summary = adaptive.summary_json();
        }

        n_past += accepted + 1;
        id_last = next;
        ++metrics.n_cycles;

        const int64_t rollback_start = ggml_time_us();
        if (!remove_after(ctx, n_past) || !remove_after(ctx_dft, n_past)) {
            LOG_ERR("failed to roll back rejected MTP verification state\n");
            return false;
        }
        metrics.rollback_us += ggml_time_us() - rollback_start;

    }

    metrics.total_us = ggml_time_us() - start;
    return true;
}

static void print_result(
        const common_params & params,
        const vegas_options & options,
        const vegas_metrics & metrics) {
    const double seconds = metrics.total_us / 1e6;
    const double tps = seconds > 0.0 ? metrics.n_predict / seconds : 0.0;
    const double accept = metrics.n_drafted > 0 ?
        (double) metrics.n_accepted / metrics.n_drafted : 0.0;

    std::printf(
        "\nVEGAS_RESULT {\"mode\":\"%s\",\"model\":\"%s\","
        "\"n_prompt\":%d,\"n_predict\":%d,\"gamma\":%d,\"auto_policy\":%s,"
        "\"selection_layer\":%d,\"anchor_tokens\":%d,\"refresh_interval\":%d,"
        "\"sparse_ratio\":%.6f,"
        "\"min_tokens\":%d,\"max_tokens\":%d,"
        "\"cache_type_k\":\"%s\",\"cache_type_v\":\"%s\","
        "\"draft_cache_type_k\":\"%s\",\"draft_cache_type_v\":\"%s\","
        "\"cycles\":%d,\"drafted\":%d,\"accepted\":%d,\"rejected\":%d,\"graphs_reused\":%d,"
        "\"output_hash\":\"%016" PRIx64 "\","
        "\"accept_rate\":%.6f,\"total_ms\":%.3f,\"tokens_per_second\":%.6f,"
        "\"prompt_ms\":%.3f,\"initial_select_ms\":%.3f,\"draft_ms\":%.3f,"
        "\"verify_ms\":%.3f,\"collect_ms\":%.3f,\"sample_ms\":%.3f,"
        "\"rollback_ms\":%.3f%s%s}\n",
        mode_name(options.mode), params.model.path.c_str(),
        metrics.n_prompt, metrics.n_predict, options.gamma, options.auto_policy ? "true" : "false",
        options.selection_layer, options.anchor_tokens, options.refresh_interval,
        options.sparse_ratio,
        options.min_tokens, options.max_tokens,
        ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
        ggml_type_name(params.speculative.draft.cache_type_k),
        ggml_type_name(params.speculative.draft.cache_type_v),
        metrics.n_cycles, metrics.n_drafted, metrics.n_accepted, metrics.n_rejected, metrics.n_reused,
        metrics.output_hash,
        accept, metrics.total_us / 1e3, tps,
        metrics.prompt_us / 1e3, metrics.initial_select_us / 1e3,
        metrics.draft_us / 1e3, metrics.verify_us / 1e3,
        metrics.collect_us / 1e3, metrics.sample_us / 1e3,
        metrics.rollback_us / 1e3,
        metrics.adaptive_summary.empty() ? "" : ",",
        metrics.adaptive_summary.c_str());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    vegas_options options;
    std::vector<char *> filtered;
    if (!parse_vegas_options(argc, argv, options, filtered)) {
        return 1;
    }

    common_params params;
    params.sampling.backend_sampling = false;

    common_init();
    if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict == 0 || params.n_predict < -1) {
        LOG_ERR("--n-predict must be -1 or greater than zero\n");
        return 1;
    }
    if (params.n_parallel != 1) {
        LOG_ERR("Vegas requires --parallel 1\n");
        return 1;
    }
    if (mode_uses_vegas(options.mode) &&
            params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LOG_ERR("Vegas requires flash attention\n");
        return 1;
    }
    if (params.sampling.mirostat != 0 || params.sampling.xtc_probability != 0.0f) {
        LOG_ERR("Vegas does not support stateful or randomized probability transforms\n");
        return 1;
    }
    if (options.adaptive_gamma && options.mode != vegas_run_mode::mtp_vegas) {
        LOG_ERR("--vegas-adaptive-gamma requires --vegas-mode mtp-vegas\n");
        return 1;
    }

    params.sampling.backend_sampling = false;
    if (options.mode == vegas_run_mode::mtp_auto && params.speculative.has_dft()) {
        options.gamma = 1;
    }
    if (options.mode != vegas_run_mode::baseline) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = options.gamma;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    if (!init) {
        return 1;
    }

    llama_model * model = init->model();
    llama_context * ctx = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, true, true);
    if (prompt.size() < 2) {
        LOG_ERR("prompt must contain at least two tokens\n");
        return 1;
    }
    if (options.prompt_tokens > 0 && options.prompt_tokens != (int32_t) prompt.size()) {
        std::vector<llama_token> resized;
        resized.reserve(options.prompt_tokens);
        resized.push_back(prompt.front());
        for (int32_t i = 1; i < options.prompt_tokens; ++i) {
            resized.push_back(prompt[1 + (i - 1) % (prompt.size() - 1)]);
        }
        prompt = std::move(resized);
    }

    resolve_auto_policy(options, params, model, (int32_t) prompt.size());
    const int32_t draft_capacity = options.adaptive_gamma ? vegas_adaptive_gamma::max_gamma : options.gamma;
    params.speculative.draft.n_max = draft_capacity;

    common_speculative_init_result_ptr spec_init;
    common_speculative_ptr spec;
    llama_context * ctx_dft = nullptr;
    if (mode_uses_mtp(options.mode)) {
        common_params params_dft = common_base_params_to_speculative(params);
        params_dft.n_ubatch = std::min(params_dft.n_ubatch, options.mtp_ubatch);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        ctx_dft = spec_init->context();
        if (ctx_dft == nullptr) {
            LOG_ERR("failed to create MTP context\n");
            return 1;
        }

        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = ctx_dft;
        spec.reset(common_speculative_init(params.speculative, 1));
        if (!spec) {
            LOG_ERR("failed to initialize MTP drafting\n");
            return 1;
        }
    }

    if (options.mode != vegas_run_mode::baseline) {
        const auto rm_type = common_context_can_seq_rm(ctx);
        if (rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART && rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
            LOG_ERR("model context does not support bounded speculative rollback\n");
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    if (ctx_dft != nullptr) {
        const auto rm_type = common_context_can_seq_rm(ctx_dft);
        if (rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART && rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
            LOG_ERR("MTP context does not support bounded speculative rollback\n");
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx_dft), true);

    }

    if (mode_uses_vegas(options.mode) &&
            !llama_vegas_enable(
                    ctx, options.sparse_ratio, options.min_tokens, options.max_tokens, draft_capacity)) {
        LOG_ERR("failed to enable Vegas\n");
        return 1;
    }

    if (options.mode == vegas_run_mode::mtp_vegas &&
            !llama_vegas_enable(
                    ctx_dft, options.sparse_ratio, options.min_tokens, options.max_tokens,
                    draft_capacity * options.refresh_interval)) {
        LOG_ERR("failed to enable Vegas for MTP context\n");
        return 1;
    }

    if (mode_uses_vegas(options.mode) &&
            (!llama_vegas_set_anchor_tokens(ctx, options.anchor_tokens) ||
             (ctx_dft != nullptr && !llama_vegas_set_anchor_tokens(ctx_dft, options.anchor_tokens)))) {
        LOG_ERR("failed to set Vegas anchor tokens\n");
        return 1;
    }

    const int32_t selection_layer = options.selection_layer >= 0 ?
            options.selection_layer : llama_model_n_layer(model) - 1;
    if (options.mode == vegas_run_mode::mtp_vegas) {
        options.selection_layer = selection_layer;
        if (!llama_vegas_set_selection_layer(ctx, selection_layer)) {
            LOG_ERR("failed to set Vegas target selection layer\n");
            return 1;
        }
    }

    if (prompt.size() + draft_capacity + 1 > llama_n_ctx(ctx)) {
        LOG_ERR("prompt and draft exceed the context size\n");
        return 1;
    }

    vegas_metrics metrics;
    metrics.n_prompt = (int32_t) prompt.size();
    const int64_t prompt_start = ggml_time_us();

    if (!decode_prompt(ctx, prompt, (int32_t) prompt.size() - 1, spec.get())) {
        LOG_ERR("failed to decode prompt\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(std::max((int32_t) llama_n_batch(ctx), draft_capacity + 1), 0, 1);
    const int32_t last_pos = (int32_t) prompt.size() - 1;

    if (mode_uses_vegas(options.mode)) {
        llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_VERIFY, (int32_t) prompt.size());
    }
    if (!decode_one(ctx, batch, prompt.back(), last_pos)) {
        LOG_ERR("failed to decode final prompt token\n");
        return 1;
    }
    llama_synchronize(ctx);
    metrics.prompt_us = ggml_time_us() - prompt_start;

    if (mode_uses_vegas(options.mode)) {
        const int64_t select_start = ggml_time_us();
        if (!llama_vegas_collect_indices(ctx)) {
            LOG_ERR("failed to initialize Vegas indices\n");
            return 1;
        }
        metrics.initial_select_us = ggml_time_us() - select_start;
    }

    if (!common_speculative_process(spec.get(), batch)) {
        LOG_ERR("failed to process final prompt token for MTP\n");
        return 1;
    }

    llama_tokens history(prompt.begin(), prompt.end());
    common_speculative_begin(spec.get(), 0, history);

    common_sampler_ptr sampler(common_sampler_init(model, params.sampling));
    llama_token id_last = common_sampler_sample(sampler.get(), ctx, 0, true);
    common_sampler_accept(sampler.get(), id_last, true);
    history.push_back(id_last);
    record_token(ctx, id_last, options.quiet, metrics);
    metrics.n_predict = 1;

    bool ok;
    if (options.mode == vegas_run_mode::baseline) {
        ok = run_baseline(
                ctx, vocab, sampler.get(), id_last, (int32_t) prompt.size(),
                params, options, batch, metrics);
    } else if (mode_uses_mtp(options.mode)) {
        ok = run_mtp(
                ctx, ctx_dft, spec.get(), vocab, sampler.get(), id_last, (int32_t) prompt.size(),
                params, options, batch, history, metrics);
    } else {
        ok = run_speculative(
                ctx, vocab, sampler.get(), id_last, (int32_t) prompt.size(),
                params, options, batch, metrics);
    }

    if (!ok) {
        llama_batch_free(batch);
        return 1;
    }

    metrics.n_reused = llama_perf_context(ctx).n_reused;
    print_result(params, options, metrics);
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
