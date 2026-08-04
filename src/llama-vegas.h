#pragma once

#include <cstdint>
#include <vector>

enum class llama_vegas_mode : int32_t {
    disabled = 0,
    draft    = 1,
    verify   = 2,
};

struct llama_vegas_state {
    llama_vegas_mode mode = llama_vegas_mode::disabled;

    float   sparse_ratio = 0.07f;
    int32_t min_tokens   = 256;
    int32_t max_tokens   = 0;
    int32_t prefix_len   = 0;
    int32_t top_k        = 0;
    int32_t max_recent_tokens = 0;
    int32_t selection_layer   = -1;

    std::vector<std::vector<int32_t>> indices;
};
