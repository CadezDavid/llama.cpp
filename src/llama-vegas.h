#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct ggml_tensor;

enum class llama_vegas_mode : int32_t {
    disabled = 0,
    draft    = 1,
    verify   = 2,
};

enum class llama_vegas_sparse_kernel : int32_t {
    direct = 0,
    gather = 1,
    auto_select = 2,
};

struct llama_vegas_state {
    llama_vegas_mode mode = llama_vegas_mode::disabled;
    llama_vegas_sparse_kernel sparse_kernel = llama_vegas_sparse_kernel::auto_select;

    float   sparse_ratio = 0.07f;
    int32_t min_tokens   = 256;
    int32_t max_tokens   = 0;
    int32_t prefix_len   = 0;
    int32_t top_k        = 0;
    int32_t max_recent_tokens = 0;
    int32_t selection_layer   = -1;
    int32_t anchor_tokens     = 0;
    ggml_tensor * plan = nullptr;
    int32_t plan_capacity = 0;
    int32_t shared_plan_layer = -1;
    ggml_backend_event_t ready_event = nullptr;
    bool wait_for_plan = false;

    std::vector<uint8_t> plan_valid;
};
