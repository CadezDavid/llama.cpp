#pragma once

#include <cstdint>

enum class vegas_hierarchical_stop {
    continue_drafting,
    eog,
    hard_cap,
    correction_cap,
    target_length,
    round_cap,
    empty_draft,
    output_limit,
};

struct vegas_hierarchical_limits {
    int32_t target_tokens = 8;
    int32_t max_tokens = 10;
    int32_t max_rounds = 3;
    int32_t max_corrections = 2;
};

// Returns the number of input rows whose KV state remains valid after inspecting
// a batched sparse verification result. A correction consumes the mismatching
// input row; an extension consumes the extra input row after the MTP prefix.
// Returns -1 for an impossible state.
int32_t vegas_hierarchical_valid_batch_inputs(
        int32_t accepted_draft_tokens,
        bool correction,
        bool extension);

vegas_hierarchical_stop vegas_hierarchical_should_stop(
        const vegas_hierarchical_limits & limits,
        int32_t provisional_tokens,
        int32_t rounds,
        int32_t corrections,
        bool provisional_eog,
        bool empty_draft,
        bool output_limited);

const char * vegas_hierarchical_stop_name(vegas_hierarchical_stop stop);
