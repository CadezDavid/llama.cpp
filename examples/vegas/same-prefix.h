#pragma once

#include <cstdint>
#include <vector>

struct vegas_same_prefix_distribution {
    int32_t top_token = -1;
    double entropy = 0.0;
    double top_probability = 0.0;
    double top_margin = 0.0;
};

struct vegas_same_prefix_comparison {
    vegas_same_prefix_distribution dense;
    vegas_same_prefix_distribution sparse;
    bool top1_match = false;
    int32_t dense_top_rank_in_sparse = 0;
    int32_t sparse_top_rank_in_dense = 0;
    double sparse_probability_of_dense_top1 = 0.0;
    double dense_probability_of_sparse_top1 = 0.0;
    double kl_dense_sparse = 0.0;
    double kl_sparse_dense = 0.0;
    double jensen_shannon = 0.0;
    double total_variation = 0.0;
    double top_k_overlap = 0.0;
};

vegas_same_prefix_comparison vegas_same_prefix_compare(
        const std::vector<float> & dense_logits,
        const std::vector<float> & sparse_logits,
        int32_t top_k = 10);
