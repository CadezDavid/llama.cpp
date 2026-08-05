#include "../examples/vegas/hierarchical-policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool condition, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d\n", line);
        std::abort();
    }
}

#define REQUIRE(condition) require((condition), __LINE__)

int main() {
    const vegas_hierarchical_limits limits;

    REQUIRE(vegas_hierarchical_should_stop(limits, 4, 1, 0, false, false, false) ==
            vegas_hierarchical_stop::continue_drafting);
    REQUIRE(vegas_hierarchical_should_stop(limits, 8, 2, 0, false, false, false) ==
            vegas_hierarchical_stop::target_length);
    REQUIRE(vegas_hierarchical_should_stop(limits, 10, 2, 0, false, false, false) ==
            vegas_hierarchical_stop::hard_cap);
    REQUIRE(vegas_hierarchical_should_stop(limits, 6, 2, 2, false, false, false) ==
            vegas_hierarchical_stop::correction_cap);
    REQUIRE(vegas_hierarchical_should_stop(limits, 6, 3, 1, false, false, false) ==
            vegas_hierarchical_stop::round_cap);
    REQUIRE(vegas_hierarchical_should_stop(limits, 2, 1, 0, true, true, true) ==
            vegas_hierarchical_stop::eog);
    REQUIRE(vegas_hierarchical_should_stop(limits, 2, 1, 0, false, true, true) ==
            vegas_hierarchical_stop::empty_draft);
    REQUIRE(vegas_hierarchical_should_stop(limits, 2, 1, 0, false, false, true) ==
            vegas_hierarchical_stop::output_limit);
    REQUIRE(std::strcmp(vegas_hierarchical_stop_name(vegas_hierarchical_stop::target_length),
                        "target_length") == 0);

    return 0;
}
