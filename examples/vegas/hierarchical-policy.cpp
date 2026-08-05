#include "hierarchical-policy.h"

int32_t vegas_hierarchical_valid_batch_inputs(
        int32_t accepted_draft_tokens,
        bool correction,
        bool extension) {
    if (accepted_draft_tokens < 0 || (correction && extension)) {
        return -1;
    }
    return accepted_draft_tokens + (correction || extension ? 1 : 0);
}

vegas_hierarchical_stop vegas_hierarchical_should_stop(
        const vegas_hierarchical_limits & limits,
        int32_t provisional_tokens,
        int32_t rounds,
        int32_t corrections,
        bool provisional_eog,
        bool empty_draft,
        bool output_limited) {
    if (provisional_eog) {
        return vegas_hierarchical_stop::eog;
    }
    if (provisional_tokens >= limits.max_tokens) {
        return vegas_hierarchical_stop::hard_cap;
    }
    if (corrections >= limits.max_corrections) {
        return vegas_hierarchical_stop::correction_cap;
    }
    if (provisional_tokens >= limits.target_tokens) {
        return vegas_hierarchical_stop::target_length;
    }
    if (rounds >= limits.max_rounds) {
        return vegas_hierarchical_stop::round_cap;
    }
    if (empty_draft) {
        return vegas_hierarchical_stop::empty_draft;
    }
    if (output_limited) {
        return vegas_hierarchical_stop::output_limit;
    }
    return vegas_hierarchical_stop::continue_drafting;
}

const char * vegas_hierarchical_stop_name(vegas_hierarchical_stop stop) {
    switch (stop) {
        case vegas_hierarchical_stop::continue_drafting: return "continue";
        case vegas_hierarchical_stop::eog:               return "eog";
        case vegas_hierarchical_stop::hard_cap:          return "hard_cap";
        case vegas_hierarchical_stop::correction_cap:    return "correction_cap";
        case vegas_hierarchical_stop::target_length:     return "target_length";
        case vegas_hierarchical_stop::round_cap:         return "round_cap";
        case vegas_hierarchical_stop::empty_draft:       return "empty_draft";
        case vegas_hierarchical_stop::output_limit:      return "output_limit";
    }
    return "unknown";
}
