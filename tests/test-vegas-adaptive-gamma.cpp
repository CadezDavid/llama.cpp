#include "../examples/vegas/adaptive-gamma.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static common_speculative_draft_observation observation(int32_t position, float entropy, float top_probability) {
    return {
        /* .position        = */ position,
        /* .top_probability = */ top_probability,
        /* .entropy         = */ entropy,
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
    vegas_adaptive_gamma policy(0.9);

    policy.begin_cycle();
    REQUIRE(policy.planned_gamma() == 10);
    for (int32_t position = 1; position <= 10; ++position) {
        const float entropy = position < 10 ? 0.1f : 2.0f;
        const float top_probability = position < 10 ? 0.95f : 0.1f;
        REQUIRE(policy.observe_draft(observation(position, entropy, top_probability)) == (position < 10));
    }
    policy.finish_cycle(10, 9, 1000, 1000);

    policy.begin_cycle();
    REQUIRE(policy.best_decision().gamma == 9);
    int32_t low_confidence_drafted = 0;
    for (int32_t position = 1; position <= 10; ++position) {
        low_confidence_drafted = position;
        if (!policy.observe_draft(observation(position, 2.0f, 0.1f))) {
            break;
        }
    }
    REQUIRE(low_confidence_drafted == 2);
    policy.finish_cycle(low_confidence_drafted, 0, 200, 1000);

    policy.begin_cycle();
    int32_t drafted = 0;
    for (int32_t position = 1; position <= 10; ++position) {
        drafted = position;
        if (!policy.observe_draft(observation(position, 0.1f, 0.95f))) {
            break;
        }
    }
    REQUIRE(drafted >= 2);
    REQUIRE(drafted <= 9);

    const auto summary = policy.summary_json();
    REQUIRE(summary.find("\"adaptive_gamma\":true") != std::string::npos);
    REQUIRE(summary.find("\"adaptive_gamma_histogram\":[0,1,0,0,0,0,0,0,0,1]") != std::string::npos);

    return 0;
}
