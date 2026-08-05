#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "adaptive-gamma.h"
#include "hierarchical-policy.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

enum class vegas_run_mode {
    baseline,
    dense_spec,
    vegas,
    mtp,
    mtp_vegas,
    mtp_hierarchical,
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
    vegas_hierarchical_limits hierarchical;
    bool hierarchical_trace = false;
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
    std::string hierarchical_summary;
};

enum class hierarchical_token_source : int32_t {
    mtp = 0,
    sparse_correction = 1,
    sparse_extension = 2,
};

struct hierarchical_entropy_stats {
    double entropy_sum = 0.0;
    double top_probability_sum = 0.0;
    int64_t samples = 0;

    void add(float entropy, float top_probability) {
        if (std::isfinite(entropy) && std::isfinite(top_probability)) {
            entropy_sum += entropy;
            top_probability_sum += top_probability;
            samples++;
        }
    }
};

struct hierarchical_round_trace {
    int32_t round = 0;
    int32_t provisional_before = 0;
    int32_t requested = 0;
    int32_t drafted = 0;
    int32_t sparse_accepted = 0;
    bool correction = false;
    bool extension = false;
    int32_t provisional_after = 0;
    int64_t mtp_us = 0;
    int64_t mtp_process_us = 0;
    int64_t sparse_us = 0;
    int64_t sparse_sample_us = 0;
    std::vector<llama_token> mtp_tokens;
    std::vector<llama_token> sparse_tokens;
    std::vector<float> mtp_entropy;
    std::vector<float> mtp_top_probability;
    std::vector<float> sparse_entropy;
    std::vector<float> sparse_top_probability;
};

struct hierarchical_cycle_trace {
    int32_t cycle = 0;
    int32_t start_pos = 0;
    vegas_hierarchical_stop stop = vegas_hierarchical_stop::continue_drafting;
    std::vector<hierarchical_round_trace> rounds;
    std::vector<llama_token> provisional_tokens;
    std::vector<int32_t> provenance;
    std::vector<uint8_t> dense_matches;
    std::vector<float> dense_entropy;
    std::vector<float> dense_top_probability;
    int32_t dense_accepted = 0;
    int32_t committed = 0;
    int32_t end_pos = 0;
    int64_t dense_us = 0;
    int64_t dense_sample_us = 0;
};

struct hierarchical_metrics {
    int64_t mtp_us = 0;
    int64_t mtp_process_us = 0;
    int64_t sparse_us = 0;
    int64_t sparse_sample_us = 0;
    int64_t dense_us = 0;
    int64_t dense_sample_us = 0;
    int64_t plan_us = 0;
    int64_t state_us = 0;
    int64_t rollback_us = 0;

    int64_t outer_cycles = 0;
    int64_t inner_rounds = 0;
    int64_t mtp_drafted = 0;
    int64_t sparse_decodes = 0;
    int64_t sparse_checked = 0;
    int64_t sparse_accepted = 0;
    int64_t sparse_corrections = 0;
    int64_t sparse_extensions = 0;
    int64_t provisional_tokens = 0;
    int64_t dense_accepted = 0;
    int64_t dense_corrections = 0;
    int64_t dense_bonus = 0;
    int64_t committed_tokens = 0;
    int64_t device_entropy_samples = 0;
    int64_t fallback_entropy_samples = 0;
    int64_t rollback_failures = 0;
    int64_t position_mismatches = 0;
    int64_t snapshot_failures = 0;
    int64_t empty_drafts = 0;

    std::array<int64_t, 4> round_histogram {};
    std::array<int64_t, 11> provisional_histogram {};
    std::array<int64_t, 11> sparse_prefix_histogram {};
    std::array<int64_t, 11> dense_prefix_histogram {};
    std::array<int64_t, 3> correction_histogram {};
    std::array<int64_t, 8> stop_histogram {};
    std::array<int64_t, 3> proposed_by_source {};
    std::array<int64_t, 3> accepted_by_source {};

    hierarchical_entropy_stats sparse_match_entropy;
    hierarchical_entropy_stats sparse_correction_entropy;
    hierarchical_entropy_stats dense_match_entropy;
    hierarchical_entropy_stats dense_rejection_entropy;

    std::vector<hierarchical_cycle_trace> traces;
};

template<typename T, size_t N>
static void json_array(std::ostringstream & stream, const std::array<T, N> & values) {
    stream << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) stream << ",";
        stream << values[i];
    }
    stream << "]";
}

template<typename T>
static void json_vector(std::ostringstream & stream, const std::vector<T> & values) {
    stream << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) stream << ",";
        stream << +values[i];
    }
    stream << "]";
}

static void json_entropy_stats(std::ostringstream & stream, const hierarchical_entropy_stats & stats) {
    stream << "{\"samples\":" << stats.samples
           << ",\"mean_entropy\":" << (stats.samples > 0 ? stats.entropy_sum / stats.samples : 0.0)
           << ",\"mean_top_probability\":"
           << (stats.samples > 0 ? stats.top_probability_sum / stats.samples : 0.0) << "}";
}

struct hierarchical_draft_observer_data {
    hierarchical_round_trace * trace = nullptr;
    hierarchical_metrics * metrics = nullptr;
};

static bool hierarchical_draft_observer(
        void * userdata,
        const common_speculative_draft_observation & observation) {
    auto * data = static_cast<hierarchical_draft_observer_data *>(userdata);
    data->trace->mtp_entropy.push_back(observation.entropy);
    data->trace->mtp_top_probability.push_back(observation.top_probability);
    if (observation.entropy_on_device) {
        data->metrics->device_entropy_samples++;
    } else {
        data->metrics->fallback_entropy_samples++;
    }
    return true;
}

struct hierarchical_entropy_observation {
    float entropy = 0.0f;
    float top_probability = 0.0f;
    bool on_device = false;
};

static hierarchical_entropy_observation hierarchical_entropy(llama_context * ctx, int32_t idx) {
    hierarchical_entropy_observation result;
    result.on_device = llama_get_sampled_entropy_ith(
            ctx, idx, &result.entropy, &result.top_probability);
    if (result.on_device) {
        return result;
    }

    const float * logits = llama_get_logits_ith(ctx, idx);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (logits == nullptr || n_vocab <= 0) {
        return result;
    }

    const float max_logit = *std::max_element(logits, logits + n_vocab);
    double sum = 0.0;
    double weighted_shifted_logit = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        const double shifted = (double) logits[token] - max_logit;
        const double weight = std::exp(shifted);
        sum += weight;
        weighted_shifted_logit += weight * shifted;
    }
    if (sum > 0.0 && std::isfinite(sum)) {
        result.top_probability = (float) (1.0 / sum);
        result.entropy = (float) (std::log(sum) - weighted_shifted_logit / sum);
    }
    return result;
}

static const char * mode_name(vegas_run_mode mode) {
    switch (mode) {
        case vegas_run_mode::baseline:   return "baseline";
        case vegas_run_mode::dense_spec: return "dense-spec";
        case vegas_run_mode::vegas:      return "vegas";
        case vegas_run_mode::mtp:        return "mtp";
        case vegas_run_mode::mtp_vegas:  return "mtp-vegas";
        case vegas_run_mode::mtp_hierarchical: return "mtp-hierarchical";
        case vegas_run_mode::mtp_auto:   return "mtp-auto";
    }
    return "unknown";
}

static bool mode_uses_vegas(vegas_run_mode mode) {
    return mode == vegas_run_mode::vegas || mode == vegas_run_mode::mtp_vegas ||
            mode == vegas_run_mode::mtp_hierarchical || mode == vegas_run_mode::mtp_auto;
}

static bool mode_uses_mtp(vegas_run_mode mode) {
    return mode == vegas_run_mode::mtp || mode == vegas_run_mode::mtp_vegas ||
            mode == vegas_run_mode::mtp_hierarchical || mode == vegas_run_mode::mtp_auto;
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
        if (arg == "--vegas-hier-trace") {
            options.hierarchical_trace = true;
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
            } else if (std::strcmp(value, "mtp-hierarchical") == 0) {
                options.mode = vegas_run_mode::mtp_hierarchical;
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
        if (const char * value = get_value("--vegas-hier-target")) {
            if (!parse_i32(value, options.hierarchical.target_tokens)) {
                LOG_ERR("invalid --vegas-hier-target: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-hier-max-tokens")) {
            if (!parse_i32(value, options.hierarchical.max_tokens)) {
                LOG_ERR("invalid --vegas-hier-max-tokens: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-hier-max-rounds")) {
            if (!parse_i32(value, options.hierarchical.max_rounds)) {
                LOG_ERR("invalid --vegas-hier-max-rounds: %s\n", value);
                return false;
            }
            continue;
        }
        if (const char * value = get_value("--vegas-hier-max-corrections")) {
            if (!parse_i32(value, options.hierarchical.max_corrections)) {
                LOG_ERR("invalid --vegas-hier-max-corrections: %s\n", value);
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
            !(options.adaptive_beta >= 0.0f && options.adaptive_beta < 1.0f) ||
            options.hierarchical.target_tokens < 1 || options.hierarchical.max_tokens < 1 ||
            options.hierarchical.target_tokens > options.hierarchical.max_tokens ||
            options.hierarchical.max_rounds < 1 || options.hierarchical.max_corrections < 1) {
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

static bool run_mtp_hierarchical(
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
        vegas_metrics & metrics,
        hierarchical_metrics & hierarchical) {
    bool has_eog = false;
    const int64_t start = ggml_time_us();

    while (!has_eog && (params.n_predict < 0 || metrics.n_predict < params.n_predict)) {
        const int32_t remaining = params.n_predict < 0 ? INT32_MAX : params.n_predict - metrics.n_predict;
        const int32_t outer_capacity = remaining > 1 ?
                std::min(options.hierarchical.max_tokens, remaining - 1) : 0;
        const int32_t outer_n_past = n_past;
        const llama_token outer_id_last = id_last;
        const size_t outer_history_size = history.size();

        if (!llama_memory_checkpoint_recurrent(ctx, 0)) {
            hierarchical.snapshot_failures++;
            LOG_ERR("failed to checkpoint target recurrent state before hierarchical cycle\n");
            return false;
        }

        hierarchical_cycle_trace cycle;
        cycle.cycle = (int32_t) hierarchical.outer_cycles;
        cycle.start_pos = outer_n_past;

        std::vector<uint8_t> spec_state;
        const int64_t snapshot_start = ggml_time_us();
        if (!common_speculative_get_state(spec, 0, spec_state)) {
            hierarchical.snapshot_failures++;
            LOG_ERR("failed to snapshot MTP state before hierarchical cycle\n");
            return false;
        }
        hierarchical.state_us += ggml_time_us() - snapshot_start;

        common_sampler_ptr provisional_sampler(common_sampler_clone(sampler));
        llama_tokens provisional_history = history;
        llama_tokens provisional;
        std::vector<hierarchical_token_source> provenance;
        provisional.reserve(outer_capacity);
        provenance.reserve(outer_capacity);
        int32_t provisional_n_past = outer_n_past;
        llama_token provisional_last = outer_id_last;
        int32_t corrections = 0;
        bool provisional_eog = false;

        if (outer_capacity > 0) {
            const int64_t plan_start = ggml_time_us();
            if (!llama_vegas_copy_indices(ctx_dft, ctx)) {
                LOG_ERR("failed to copy authoritative Vegas indices for hierarchical MTP\n");
                return false;
            }
            hierarchical.plan_us += ggml_time_us() - plan_start;
        }

        vegas_hierarchical_stop stop = outer_capacity > 0 ?
                vegas_hierarchical_stop::continue_drafting : vegas_hierarchical_stop::output_limit;

        while (stop == vegas_hierarchical_stop::continue_drafting) {
            hierarchical_round_trace round;
            round.round = (int32_t) cycle.rounds.size();
            round.provisional_before = (int32_t) provisional.size();
            round.requested = std::min(options.gamma, outer_capacity - (int32_t) provisional.size());
            if (round.requested <= 0) {
                stop = vegas_hierarchical_stop::hard_cap;
                break;
            }

            llama_tokens draft;
            if (!llama_vegas_resume_draft(ctx_dft)) {
                LOG_ERR("failed to resume hierarchical MTP Vegas plan\n");
                return false;
            }

            auto & draft_params = common_speculative_get_draft_params(spec, 0);
            draft_params = {
                /* .drafting = */ true,
                /* .n_max    = */ round.requested,
                /* .n_past   = */ provisional_n_past,
                /* .id_last  = */ provisional_last,
                /* .prompt   = */ &provisional_history,
                /* .result   = */ &draft,
            };
            hierarchical_draft_observer_data observer_data {
                /* .trace   = */ &round,
                /* .metrics = */ &hierarchical,
            };
            draft_params.observer = hierarchical_draft_observer;
            draft_params.observer_userdata = &observer_data;

            const int64_t mtp_start = ggml_time_us();
            common_speculative_draft(spec);
            llama_synchronize(ctx_dft);
            round.mtp_us = ggml_time_us() - mtp_start;
            hierarchical.mtp_us += round.mtp_us;
            hierarchical.mtp_drafted += draft.size();
            metrics.draft_us += round.mtp_us;
            metrics.n_drafted += (int32_t) draft.size();
            round.drafted = (int32_t) draft.size();
            round.mtp_tokens = draft;

            llama_vegas_pause(ctx_dft);
            const int64_t draft_rollback_start = ggml_time_us();
            if (!remove_after(ctx_dft, provisional_n_past)) {
                hierarchical.rollback_failures++;
                LOG_ERR("failed to roll back hierarchical MTP draft state\n");
                return false;
            }
            const int64_t draft_rollback_us = ggml_time_us() - draft_rollback_start;
            hierarchical.rollback_us += draft_rollback_us;
            metrics.rollback_us += draft_rollback_us;

            const bool empty_draft = draft.empty();
            if (!empty_draft) {
                if (!llama_vegas_resume_draft(ctx)) {
                    LOG_ERR("failed to resume sparse-target Vegas plan\n");
                    return false;
                }

                bool mismatch = false;
                for (size_t i = 0; i < draft.size(); ++i) {
                    const int64_t sparse_start = ggml_time_us();
                    if (!decode_one(ctx, batch, provisional_last, provisional_n_past)) {
                        return false;
                    }
                    llama_synchronize(ctx);
                    const int64_t sparse_decode_us = ggml_time_us() - sparse_start;
                    round.sparse_us += sparse_decode_us;
                    hierarchical.sparse_us += sparse_decode_us;
                    hierarchical.sparse_decodes++;

                    const auto entropy = hierarchical_entropy(ctx, 0);
                    round.sparse_entropy.push_back(entropy.entropy);
                    round.sparse_top_probability.push_back(entropy.top_probability);
                    if (entropy.on_device) hierarchical.device_entropy_samples++;
                    else hierarchical.fallback_entropy_samples++;

                    const int64_t process_start = ggml_time_us();
                    if (!common_speculative_process(spec, batch)) {
                        LOG_ERR("failed to process sparse-target token for MTP\n");
                        return false;
                    }
                    llama_synchronize(ctx_dft);
                    const int64_t process_us = ggml_time_us() - process_start;
                    round.mtp_process_us += process_us;
                    hierarchical.mtp_process_us += process_us;

                    const int64_t sample_start = ggml_time_us();
                    const llama_token sparse_token = common_sampler_sample(provisional_sampler.get(), ctx, 0, true);
                    round.sparse_sample_us += ggml_time_us() - sample_start;
                    round.sparse_tokens.push_back(sparse_token);
                    hierarchical.sparse_checked++;

                    if (sparse_token == draft[i]) {
                        common_sampler_accept(provisional_sampler.get(), draft[i], true);
                        provisional.push_back(draft[i]);
                        provenance.push_back(hierarchical_token_source::mtp);
                        provisional_history.push_back(draft[i]);
                        provisional_last = draft[i];
                        provisional_n_past++;
                        round.sparse_accepted++;
                        hierarchical.sparse_accepted++;
                        hierarchical.proposed_by_source[(int) hierarchical_token_source::mtp]++;
                        hierarchical.sparse_match_entropy.add(entropy.entropy, entropy.top_probability);
                        provisional_eog = llama_vocab_is_eog(vocab, draft[i]);
                        if (provisional_eog || (int32_t) provisional.size() >= outer_capacity) {
                            break;
                        }
                        continue;
                    }

                    common_sampler_accept(provisional_sampler.get(), sparse_token, true);
                    provisional.push_back(sparse_token);
                    provenance.push_back(hierarchical_token_source::sparse_correction);
                    provisional_history.push_back(sparse_token);
                    provisional_last = sparse_token;
                    provisional_n_past++;
                    corrections++;
                    round.correction = true;
                    hierarchical.sparse_corrections++;
                    hierarchical.proposed_by_source[(int) hierarchical_token_source::sparse_correction]++;
                    hierarchical.sparse_correction_entropy.add(entropy.entropy, entropy.top_probability);
                    provisional_eog = llama_vocab_is_eog(vocab, sparse_token);
                    mismatch = true;
                    break;
                }

                if (!mismatch && !provisional_eog && round.sparse_accepted == (int32_t) draft.size() &&
                        (int32_t) provisional.size() < outer_capacity) {
                    const int64_t sparse_start = ggml_time_us();
                    if (!decode_one(ctx, batch, provisional_last, provisional_n_past)) {
                        return false;
                    }
                    llama_synchronize(ctx);
                    const int64_t sparse_decode_us = ggml_time_us() - sparse_start;
                    round.sparse_us += sparse_decode_us;
                    hierarchical.sparse_us += sparse_decode_us;
                    hierarchical.sparse_decodes++;

                    const auto entropy = hierarchical_entropy(ctx, 0);
                    round.sparse_entropy.push_back(entropy.entropy);
                    round.sparse_top_probability.push_back(entropy.top_probability);
                    if (entropy.on_device) hierarchical.device_entropy_samples++;
                    else hierarchical.fallback_entropy_samples++;

                    const int64_t process_start = ggml_time_us();
                    if (!common_speculative_process(spec, batch)) {
                        LOG_ERR("failed to process sparse-target extension for MTP\n");
                        return false;
                    }
                    llama_synchronize(ctx_dft);
                    const int64_t process_us = ggml_time_us() - process_start;
                    round.mtp_process_us += process_us;
                    hierarchical.mtp_process_us += process_us;

                    const int64_t sample_start = ggml_time_us();
                    const llama_token extension = common_sampler_sample(provisional_sampler.get(), ctx, 0, true);
                    round.sparse_sample_us += ggml_time_us() - sample_start;
                    round.sparse_tokens.push_back(extension);
                    common_sampler_accept(provisional_sampler.get(), extension, true);
                    provisional.push_back(extension);
                    provenance.push_back(hierarchical_token_source::sparse_extension);
                    provisional_history.push_back(extension);
                    provisional_last = extension;
                    provisional_n_past++;
                    round.extension = true;
                    hierarchical.sparse_extensions++;
                    hierarchical.proposed_by_source[(int) hierarchical_token_source::sparse_extension]++;
                    provisional_eog = llama_vocab_is_eog(vocab, extension);
                }

                hierarchical.sparse_sample_us += round.sparse_sample_us;
                common_speculative_accept(spec, 0, round.sparse_accepted);
            } else {
                hierarchical.empty_drafts++;
            }

            round.provisional_after = (int32_t) provisional.size();
            cycle.rounds.push_back(std::move(round));
            hierarchical.inner_rounds++;

            stop = vegas_hierarchical_should_stop(
                    options.hierarchical,
                    (int32_t) provisional.size(),
                    (int32_t) cycle.rounds.size(),
                    corrections,
                    provisional_eog,
                    empty_draft,
                    (int32_t) provisional.size() >= outer_capacity);
        }

        cycle.stop = stop;
        cycle.provisional_tokens = provisional;
        cycle.provenance.reserve(provenance.size());
        for (auto source : provenance) cycle.provenance.push_back((int32_t) source);

        const int64_t restore_start = ggml_time_us();
        llama_vegas_pause(ctx);
        llama_vegas_pause(ctx_dft);
        const llama_pos target_pos_before_restore = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        const llama_pos draft_pos_before_restore = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0);
        const bool target_restored = remove_after(ctx, outer_n_past);
        const bool draft_restored = remove_after(ctx_dft, outer_n_past);
        if (!target_restored || !draft_restored) {
            hierarchical.rollback_failures++;
            LOG_ERR("failed to restore authoritative KV prefix before dense verification: "
                    "target_ok=%d draft_ok=%d target_pos=%d draft_pos=%d restore_pos=%d\n",
                    target_restored, draft_restored, (int) target_pos_before_restore,
                    (int) draft_pos_before_restore, outer_n_past);
            return false;
        }
        if (!llama_memory_restore_recurrent(ctx, 0)) {
            hierarchical.snapshot_failures++;
            LOG_ERR("failed to restore target recurrent checkpoint before dense verification\n");
            return false;
        }
        if (!common_speculative_set_state(spec, 0, spec_state)) {
            hierarchical.snapshot_failures++;
            LOG_ERR("failed to restore authoritative MTP state\n");
            return false;
        }
        std::vector<uint8_t> restored_state;
        if (!common_speculative_get_state(spec, 0, restored_state) || restored_state != spec_state) {
            hierarchical.snapshot_failures++;
            LOG_ERR("MTP state did not round-trip during hierarchical restore\n");
            return false;
        }
        const int64_t restore_us = ggml_time_us() - restore_start;
        hierarchical.state_us += restore_us;

        const bool refresh_indices = (metrics.n_cycles + 1) % options.refresh_interval == 0;
        if (refresh_indices) {
            llama_vegas_set_mode(ctx, LLAMA_VEGAS_MODE_VERIFY, outer_n_past + 1);
        } else {
            llama_vegas_pause(ctx);
        }

        common_batch_clear(batch);
        common_batch_add(batch, outer_id_last, outer_n_past, { 0 }, true);
        for (size_t i = 0; i < provisional.size(); ++i) {
            common_batch_add(batch, provisional[i], outer_n_past + (int32_t) i + 1, { 0 }, true);
        }

        const int64_t dense_start = ggml_time_us();
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
        llama_synchronize(ctx);
        cycle.dense_us = ggml_time_us() - dense_start;
        hierarchical.dense_us += cycle.dense_us;
        metrics.verify_us += cycle.dense_us;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const auto entropy = hierarchical_entropy(ctx, i);
            cycle.dense_entropy.push_back(entropy.entropy);
            cycle.dense_top_probability.push_back(entropy.top_probability);
            if (entropy.on_device) hierarchical.device_entropy_samples++;
            else hierarchical.fallback_entropy_samples++;
        }

        const int64_t process_start = ggml_time_us();
        if (!common_speculative_process(spec, batch)) {
            LOG_ERR("failed to process dense hierarchical verification for MTP\n");
            return false;
        }
        llama_synchronize(ctx_dft);
        hierarchical.mtp_process_us += ggml_time_us() - process_start;

        if (refresh_indices) {
            const int64_t plan_start = ggml_time_us();
            if (!llama_vegas_collect_indices(ctx)) {
                LOG_ERR("failed to collect authoritative hierarchical Vegas indices\n");
                return false;
            }
            const int64_t plan_us = ggml_time_us() - plan_start;
            hierarchical.plan_us += plan_us;
            metrics.collect_us += plan_us;
        }

        int32_t accepted = 0;
        llama_token next = LLAMA_TOKEN_NULL;
        bool emitted_dense_extra = false;
        const int64_t dense_sample_start = ggml_time_us();
        for (size_t i = 0; i < provisional.size(); ++i) {
            const llama_token target = common_sampler_sample(sampler, ctx, (int32_t) i, true);
            const bool match = target == provisional[i];
            cycle.dense_matches.push_back(match ? 1 : 0);
            const auto source = provenance[i];
            if (match) {
                next = provisional[i];
                common_sampler_accept(sampler, next, true);
                accepted++;
                metrics.n_accepted++;
                metrics.n_predict++;
                hierarchical.dense_accepted++;
                hierarchical.accepted_by_source[(int) source]++;
                hierarchical.dense_match_entropy.add(
                        cycle.dense_entropy[i], cycle.dense_top_probability[i]);
                history.push_back(next);
                record_token(ctx, next, options.quiet, metrics);
                if (llama_vocab_is_eog(vocab, next)) {
                    has_eog = true;
                    break;
                }
                continue;
            }

            next = target;
            common_sampler_accept(sampler, next, true);
            metrics.n_rejected++;
            metrics.n_predict++;
            hierarchical.dense_corrections++;
            emitted_dense_extra = true;
            hierarchical.dense_rejection_entropy.add(
                    cycle.dense_entropy[i], cycle.dense_top_probability[i]);
            history.push_back(next);
            record_token(ctx, next, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, next);
            break;
        }

        if (!has_eog && accepted == (int32_t) provisional.size()) {
            next = common_sampler_sample(sampler, ctx, (int32_t) provisional.size(), true);
            common_sampler_accept(sampler, next, true);
            metrics.n_predict++;
            hierarchical.dense_bonus++;
            emitted_dense_extra = true;
            history.push_back(next);
            record_token(ctx, next, options.quiet, metrics);
            has_eog = llama_vocab_is_eog(vocab, next);
        }
        cycle.dense_sample_us = ggml_time_us() - dense_sample_start;
        hierarchical.dense_sample_us += cycle.dense_sample_us;
        metrics.sample_us += cycle.dense_sample_us;

        common_speculative_accept(spec, 0, accepted);

        const int32_t committed = accepted + (emitted_dense_extra ? 1 : 0);
        hierarchical.committed_tokens += committed;
        hierarchical.provisional_tokens += provisional.size();
        cycle.dense_accepted = accepted;
        cycle.committed = committed;

        n_past += accepted + 1;
        id_last = next;
        metrics.n_cycles++;
        hierarchical.outer_cycles++;

        hierarchical.round_histogram[std::min<size_t>(cycle.rounds.size(), 3)]++;
        hierarchical.provisional_histogram[std::min<size_t>(provisional.size(), 10)]++;
        hierarchical.dense_prefix_histogram[std::min(accepted, 10)]++;
        hierarchical.correction_histogram[std::min(corrections, 2)]++;
        hierarchical.stop_histogram[(int) stop]++;
        for (const auto & round : cycle.rounds) {
            hierarchical.sparse_prefix_histogram[std::min(round.sparse_accepted, 10)]++;
        }

        const int64_t rollback_start = ggml_time_us();
        if (!remove_after(ctx, n_past) || !remove_after(ctx_dft, n_past)) {
            hierarchical.rollback_failures++;
            LOG_ERR("failed to roll back rejected hierarchical verification state\n");
            return false;
        }
        const int64_t rollback_us = ggml_time_us() - rollback_start;
        hierarchical.rollback_us += rollback_us;
        metrics.rollback_us += rollback_us;

        const llama_pos target_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        const llama_pos draft_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0);
        if (target_pos != n_past - 1 || draft_pos != n_past - 1 ||
                history.size() != outer_history_size + (size_t) committed) {
            hierarchical.position_mismatches++;
            LOG_ERR("hierarchical state mismatch: target=%d draft=%d expected=%d history=%zu expected_history=%zu\n",
                    (int) target_pos, (int) draft_pos, n_past - 1,
                    history.size(), outer_history_size + (size_t) committed);
            return false;
        }

        cycle.end_pos = n_past;
        if (options.hierarchical_trace) {
            hierarchical.traces.push_back(std::move(cycle));
        }
    }

    metrics.total_us = ggml_time_us() - start;
    return true;
}

static std::string hierarchical_summary_json(
        const vegas_options & options,
        const hierarchical_metrics & metrics) {
    std::ostringstream out;
    out << "\"hierarchical\":true"
        << ",\"hierarchical_trace_schema\":1"
        << ",\"hierarchical_target\":" << options.hierarchical.target_tokens
        << ",\"hierarchical_max_tokens\":" << options.hierarchical.max_tokens
        << ",\"hierarchical_max_rounds\":" << options.hierarchical.max_rounds
        << ",\"hierarchical_max_corrections\":" << options.hierarchical.max_corrections
        << ",\"hierarchical_outer_cycles\":" << metrics.outer_cycles
        << ",\"hierarchical_inner_rounds\":" << metrics.inner_rounds
        << ",\"hierarchical_mtp_drafted\":" << metrics.mtp_drafted
        << ",\"hierarchical_sparse_decodes\":" << metrics.sparse_decodes
        << ",\"hierarchical_sparse_checked\":" << metrics.sparse_checked
        << ",\"hierarchical_sparse_accepted\":" << metrics.sparse_accepted
        << ",\"hierarchical_sparse_corrections\":" << metrics.sparse_corrections
        << ",\"hierarchical_sparse_extensions\":" << metrics.sparse_extensions
        << ",\"hierarchical_provisional_tokens\":" << metrics.provisional_tokens
        << ",\"hierarchical_dense_accepted\":" << metrics.dense_accepted
        << ",\"hierarchical_dense_corrections\":" << metrics.dense_corrections
        << ",\"hierarchical_dense_bonus\":" << metrics.dense_bonus
        << ",\"hierarchical_committed_tokens\":" << metrics.committed_tokens
        << ",\"hierarchical_committed_per_dense_cycle\":"
        << (metrics.outer_cycles > 0 ? (double) metrics.committed_tokens / metrics.outer_cycles : 0.0)
        << ",\"hierarchical_sparse_agreement\":"
        << (metrics.sparse_checked > 0 ? (double) metrics.sparse_accepted / metrics.sparse_checked : 0.0)
        << ",\"hierarchical_dense_acceptance\":"
        << (metrics.provisional_tokens > 0 ? (double) metrics.dense_accepted / metrics.provisional_tokens : 0.0)
        << ",\"hierarchical_mtp_ms\":" << metrics.mtp_us / 1e3
        << ",\"hierarchical_mtp_process_ms\":" << metrics.mtp_process_us / 1e3
        << ",\"hierarchical_sparse_ms\":" << metrics.sparse_us / 1e3
        << ",\"hierarchical_sparse_sample_ms\":" << metrics.sparse_sample_us / 1e3
        << ",\"hierarchical_dense_ms\":" << metrics.dense_us / 1e3
        << ",\"hierarchical_dense_sample_ms\":" << metrics.dense_sample_us / 1e3
        << ",\"hierarchical_plan_ms\":" << metrics.plan_us / 1e3
        << ",\"hierarchical_state_ms\":" << metrics.state_us / 1e3
        << ",\"hierarchical_rollback_ms\":" << metrics.rollback_us / 1e3
        << ",\"hierarchical_device_entropy_samples\":" << metrics.device_entropy_samples
        << ",\"hierarchical_fallback_entropy_samples\":" << metrics.fallback_entropy_samples
        << ",\"hierarchical_rollback_failures\":" << metrics.rollback_failures
        << ",\"hierarchical_position_mismatches\":" << metrics.position_mismatches
        << ",\"hierarchical_snapshot_failures\":" << metrics.snapshot_failures
        << ",\"hierarchical_empty_drafts\":" << metrics.empty_drafts;

    out << ",\"hierarchical_round_histogram\":";
    json_array(out, metrics.round_histogram);
    out << ",\"hierarchical_provisional_histogram\":";
    json_array(out, metrics.provisional_histogram);
    out << ",\"hierarchical_sparse_prefix_histogram\":";
    json_array(out, metrics.sparse_prefix_histogram);
    out << ",\"hierarchical_dense_prefix_histogram\":";
    json_array(out, metrics.dense_prefix_histogram);
    out << ",\"hierarchical_correction_histogram\":";
    json_array(out, metrics.correction_histogram);
    out << ",\"hierarchical_stop_histogram\":";
    json_array(out, metrics.stop_histogram);
    out << ",\"hierarchical_proposed_by_source\":";
    json_array(out, metrics.proposed_by_source);
    out << ",\"hierarchical_accepted_by_source\":";
    json_array(out, metrics.accepted_by_source);

    out << ",\"hierarchical_sparse_match_entropy\":";
    json_entropy_stats(out, metrics.sparse_match_entropy);
    out << ",\"hierarchical_sparse_correction_entropy\":";
    json_entropy_stats(out, metrics.sparse_correction_entropy);
    out << ",\"hierarchical_dense_match_entropy\":";
    json_entropy_stats(out, metrics.dense_match_entropy);
    out << ",\"hierarchical_dense_rejection_entropy\":";
    json_entropy_stats(out, metrics.dense_rejection_entropy);

    if (options.hierarchical_trace) {
        out << ",\"hierarchical_trace\":[";
        for (size_t i = 0; i < metrics.traces.size(); ++i) {
            if (i > 0) out << ",";
            const auto & cycle = metrics.traces[i];
            out << "{\"cycle\":" << cycle.cycle
                << ",\"start_pos\":" << cycle.start_pos
                << ",\"stop\":\"" << vegas_hierarchical_stop_name(cycle.stop) << "\""
                << ",\"provisional_tokens\":";
            json_vector(out, cycle.provisional_tokens);
            out << ",\"provenance\":";
            json_vector(out, cycle.provenance);
            out << ",\"dense_matches\":";
            json_vector(out, cycle.dense_matches);
            out << ",\"dense_entropy\":";
            json_vector(out, cycle.dense_entropy);
            out << ",\"dense_top_probability\":";
            json_vector(out, cycle.dense_top_probability);
            out << ",\"dense_accepted\":" << cycle.dense_accepted
                << ",\"committed\":" << cycle.committed
                << ",\"end_pos\":" << cycle.end_pos
                << ",\"dense_ms\":" << cycle.dense_us / 1e3
                << ",\"dense_sample_ms\":" << cycle.dense_sample_us / 1e3
                << ",\"rounds\":[";
            for (size_t j = 0; j < cycle.rounds.size(); ++j) {
                if (j > 0) out << ",";
                const auto & round = cycle.rounds[j];
                out << "{\"round\":" << round.round
                    << ",\"provisional_before\":" << round.provisional_before
                    << ",\"requested\":" << round.requested
                    << ",\"drafted\":" << round.drafted
                    << ",\"sparse_accepted\":" << round.sparse_accepted
                    << ",\"correction\":" << (round.correction ? "true" : "false")
                    << ",\"extension\":" << (round.extension ? "true" : "false")
                    << ",\"provisional_after\":" << round.provisional_after
                    << ",\"mtp_ms\":" << round.mtp_us / 1e3
                    << ",\"mtp_process_ms\":" << round.mtp_process_us / 1e3
                    << ",\"sparse_ms\":" << round.sparse_us / 1e3
                    << ",\"sparse_sample_ms\":" << round.sparse_sample_us / 1e3
                    << ",\"mtp_tokens\":";
                json_vector(out, round.mtp_tokens);
                out << ",\"sparse_tokens\":";
                json_vector(out, round.sparse_tokens);
                out << ",\"mtp_entropy\":";
                json_vector(out, round.mtp_entropy);
                out << ",\"mtp_top_probability\":";
                json_vector(out, round.mtp_top_probability);
                out << ",\"sparse_entropy\":";
                json_vector(out, round.sparse_entropy);
                out << ",\"sparse_top_probability\":";
                json_vector(out, round.sparse_top_probability);
                out << "}";
            }
            out << "]}";
        }
        out << "]";
    }

    return out.str();
}

static void print_result(
        const common_params & params,
        const vegas_options & options,
        const vegas_metrics & metrics) {
    const double seconds = metrics.total_us / 1e6;
    const double tps = seconds > 0.0 ? metrics.n_predict / seconds : 0.0;
    const double accept = metrics.n_drafted > 0 ?
        (double) metrics.n_accepted / metrics.n_drafted : 0.0;
    std::string extra;
    if (!metrics.adaptive_summary.empty()) {
        extra += "," + metrics.adaptive_summary;
    }
    if (!metrics.hierarchical_summary.empty()) {
        extra += "," + metrics.hierarchical_summary;
    }

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
        "\"rollback_ms\":%.3f%s}\n",
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
        metrics.rollback_us / 1e3, extra.c_str());
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
    if (options.hierarchical_trace && options.mode != vegas_run_mode::mtp_hierarchical) {
        LOG_ERR("--vegas-hier-trace requires --vegas-mode mtp-hierarchical\n");
        return 1;
    }

    params.sampling.backend_sampling = false;
    if (options.mode == vegas_run_mode::mtp_auto && params.speculative.has_dft()) {
        options.gamma = 1;
    }
    if (options.mode != vegas_run_mode::baseline) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = options.mode == vegas_run_mode::mtp_hierarchical ?
                options.hierarchical.max_tokens :
                (options.adaptive_gamma ? vegas_adaptive_gamma::max_gamma : options.gamma);
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
    const int32_t draft_capacity = options.mode == vegas_run_mode::mtp_hierarchical ?
            options.hierarchical.max_tokens :
            (options.adaptive_gamma ? vegas_adaptive_gamma::max_gamma : options.gamma);
    // Include both MTP boundary positions around the verification and draft
    // spans. Qwen MoE can reach both extras when an inner round starts after a
    // fully accepted block.
    const int32_t recent_capacity = (options.refresh_interval + 1) * (draft_capacity + 1) + 2;
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
                    ctx, options.sparse_ratio, options.min_tokens, options.max_tokens, recent_capacity)) {
        LOG_ERR("failed to enable Vegas\n");
        return 1;
    }

    if ((options.mode == vegas_run_mode::mtp_vegas || options.mode == vegas_run_mode::mtp_hierarchical) &&
            !llama_vegas_enable(
                    ctx_dft, options.sparse_ratio, options.min_tokens, options.max_tokens,
                    recent_capacity)) {
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
    if (options.mode == vegas_run_mode::mtp_vegas || options.mode == vegas_run_mode::mtp_hierarchical) {
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

    if (options.mode == vegas_run_mode::mtp_hierarchical) {
        llama_set_entropy_output(ctx, true);
    }

    bool ok;
    hierarchical_metrics hierarchical;
    if (options.mode == vegas_run_mode::baseline) {
        ok = run_baseline(
                ctx, vocab, sampler.get(), id_last, (int32_t) prompt.size(),
                params, options, batch, metrics);
    } else if (options.mode == vegas_run_mode::mtp_hierarchical) {
        ok = run_mtp_hierarchical(
                ctx, ctx_dft, spec.get(), vocab, sampler.get(), id_last, (int32_t) prompt.size(),
                params, options, batch, history, metrics, hierarchical);
        metrics.hierarchical_summary = hierarchical_summary_json(options, hierarchical);
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

    if (options.mode == vegas_run_mode::mtp_hierarchical) {
        llama_set_entropy_output(ctx, false);
    }
    metrics.n_reused = llama_perf_context(ctx).n_reused;
    print_result(params, options, metrics);
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
