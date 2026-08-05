#include "../examples/vegas/adaptive-gamma.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static common_speculative_draft_observation observation(int32_t position, float entropy, float top_probability) {
    return {
        /* .position        = */ position,
        /* .top_probability = */ top_probability,
        /* .entropy         = */ entropy,
        /* .entropy_on_device = */ true,
    };
}

static void require(bool condition, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d\n", line);
        std::abort();
    }
}

#define REQUIRE(condition) require((condition), __LINE__)

int main() {
    vegas_adaptive_gamma policy(0.9, 4);

    // Cold start with the configured dense-MTP horizon, not gamma 10.
    policy.begin_cycle();
    REQUIRE(policy.planned_gamma() == 4);
    REQUIRE(!policy.planned_sparse());
    for (int32_t position = 1; position <= 4; ++position) {
        REQUIRE(policy.observe_draft(observation(position, 0.1f, 0.95f)) == (position < 4));
    }
    policy.finish_cycle(4, 4, 400, 1000, false);

    // Measure the same safe horizon once with Vegas before comparing modes.
    policy.begin_cycle();
    REQUIRE(policy.planned_gamma() == 4);
    REQUIRE(policy.planned_sparse());
    for (int32_t position = 1; position <= 4; ++position) {
        REQUIRE(policy.observe_draft(observation(position, 0.1f, 0.95f)) == (position < 4));
    }
    policy.finish_cycle(4, 4, 200, 1000, true);

    policy.begin_cycle();
    REQUIRE(policy.planned_sparse());
    int32_t low_confidence_drafted = 0;
    for (int32_t position = 1; position <= 10; ++position) {
        low_confidence_drafted = position;
        if (!policy.observe_draft(observation(position, 2.0f, 0.1f))) {
            break;
        }
    }
    REQUIRE(low_confidence_drafted >= 1);
    REQUIRE(low_confidence_drafted < 4);
    policy.finish_cycle(low_confidence_drafted, 0, 200, 1000, true);

    const auto summary = policy.summary_json();
    REQUIRE(summary.find("\"adaptive_gamma\":true") != std::string::npos);
    REQUIRE(summary.find("\"adaptive_dense_cycles\":1") != std::string::npos);
    REQUIRE(summary.find("\"adaptive_sparse_cycles\":2") != std::string::npos);
    REQUIRE(summary.find("\"adaptive_fallback_entropy_samples\":0") != std::string::npos);

    vegas_adaptive_gamma dense_fallback(0.9, 3);
    dense_fallback.begin_cycle();
    for (int32_t position = 1; position <= 3; ++position) {
        dense_fallback.observe_draft(observation(position, 0.1f, 0.95f));
    }
    dense_fallback.finish_cycle(3, 3, 100, 300, false);

    dense_fallback.begin_cycle();
    for (int32_t position = 1; position <= 3; ++position) {
        dense_fallback.observe_draft(observation(position, 2.0f, 0.1f));
    }
    dense_fallback.finish_cycle(3, 0, 300, 500, true);

    dense_fallback.begin_cycle();
    REQUIRE(!dense_fallback.planned_sparse());
    REQUIRE(dense_fallback.planned_gamma() == 3);

    return 0;
}
