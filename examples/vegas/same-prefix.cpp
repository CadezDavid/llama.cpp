#include "same-prefix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <queue>
#include <utility>

namespace {

struct normalized_distribution {
    vegas_same_prefix_distribution summary;
    std::vector<double> probabilities;
    std::vector<double> log_probabilities;
};

normalized_distribution normalize(const std::vector<float> & logits) {
    assert(!logits.empty());

    normalized_distribution result;
    result.probabilities.resize(logits.size());
    result.log_probabilities.resize(logits.size());

    int32_t top = 0;
    int32_t second = logits.size() > 1 ? 1 : 0;
    if (logits.size() > 1 && logits[second] > logits[top]) {
        std::swap(top, second);
    }
    for (int32_t i = 2; i < (int32_t) logits.size(); ++i) {
        if (logits[i] > logits[top]) {
            second = top;
            top = i;
        } else if (logits[i] > logits[second]) {
            second = i;
        }
    }

    const double max_logit = logits[top];
    double sum = 0.0;
    for (float logit : logits) {
        sum += std::exp((double) logit - max_logit);
    }
    const double log_normalizer = max_logit + std::log(sum);

    double entropy = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double log_probability = (double) logits[i] - log_normalizer;
        const double probability = std::exp(log_probability);
        result.log_probabilities[i] = log_probability;
        result.probabilities[i] = probability;
        entropy -= probability * log_probability;
    }

    result.summary.top_token = top;
    result.summary.entropy = entropy;
    result.summary.top_probability = result.probabilities[top];
    result.summary.top_margin = logits.size() > 1 ? (double) logits[top] - logits[second] : 0.0;
    return result;
}

int32_t rank_of(const std::vector<float> & logits, int32_t token) {
    int32_t rank = 1;
    for (int32_t i = 0; i < (int32_t) logits.size(); ++i) {
        if (logits[i] > logits[token] || (logits[i] == logits[token] && i < token)) {
            rank++;
        }
    }
    return rank;
}

std::vector<int32_t> top_tokens(const std::vector<float> & logits, int32_t top_k) {
    using entry = std::pair<float, int32_t>;
    std::priority_queue<entry, std::vector<entry>, std::greater<entry>> heap;
    for (int32_t token = 0; token < (int32_t) logits.size(); ++token) {
        const entry candidate { logits[token], -token };
        if ((int32_t) heap.size() < top_k) {
            heap.push(candidate);
        } else if (candidate > heap.top()) {
            heap.pop();
            heap.push(candidate);
        }
    }

    std::vector<int32_t> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(-heap.top().second);
        heap.pop();
    }
    return result;
}

} // namespace

vegas_same_prefix_comparison vegas_same_prefix_compare(
        const std::vector<float> & dense_logits,
        const std::vector<float> & sparse_logits,
        int32_t top_k) {
    assert(!dense_logits.empty());
    assert(dense_logits.size() == sparse_logits.size());
    assert(top_k > 0);

    const auto dense = normalize(dense_logits);
    const auto sparse = normalize(sparse_logits);

    vegas_same_prefix_comparison result;
    result.dense = dense.summary;
    result.sparse = sparse.summary;
    result.top1_match = dense.summary.top_token == sparse.summary.top_token;
    result.dense_top_rank_in_sparse = rank_of(sparse_logits, dense.summary.top_token);
    result.sparse_top_rank_in_dense = rank_of(dense_logits, sparse.summary.top_token);
    result.sparse_probability_of_dense_top1 = sparse.probabilities[dense.summary.top_token];
    result.dense_probability_of_sparse_top1 = dense.probabilities[sparse.summary.top_token];

    for (size_t i = 0; i < dense_logits.size(); ++i) {
        const double p = dense.probabilities[i];
        const double q = sparse.probabilities[i];
        const double log_p = dense.log_probabilities[i];
        const double log_q = sparse.log_probabilities[i];
        const double mixture = 0.5 * (p + q);

        if (p > 0.0) {
            result.kl_dense_sparse += p * (log_p - log_q);
        }
        if (q > 0.0) {
            result.kl_sparse_dense += q * (log_q - log_p);
        }
        if (mixture > 0.0) {
            if (p > 0.0) {
                result.jensen_shannon += 0.5 * p * (log_p - std::log(mixture));
            }
            if (q > 0.0) {
                result.jensen_shannon += 0.5 * q * (log_q - std::log(mixture));
            }
        }
        result.total_variation += 0.5 * std::abs(p - q);
    }

    result.kl_dense_sparse = std::max(0.0, result.kl_dense_sparse);
    result.kl_sparse_dense = std::max(0.0, result.kl_sparse_dense);
    result.jensen_shannon = std::max(0.0, result.jensen_shannon);

    const int32_t effective_top_k = std::min<int32_t>(top_k, dense_logits.size());
    const auto dense_top = top_tokens(dense_logits, effective_top_k);
    const auto sparse_top = top_tokens(sparse_logits, effective_top_k);
    int32_t overlap = 0;
    for (int32_t token : dense_top) {
        overlap += std::find(sparse_top.begin(), sparse_top.end(), token) != sparse_top.end();
    }
    result.top_k_overlap = (double) overlap / effective_top_k;

    return result;
}
