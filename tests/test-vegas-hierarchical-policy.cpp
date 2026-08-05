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

    // A mismatch at draft position i leaves the i matched input rows plus the
    // mismatching row valid. These cover first, middle, and last-token repairs.
    REQUIRE(vegas_hierarchical_valid_batch_inputs(0, true, false) == 1);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(2, true, false) == 3);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(8, true, false) == 9);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(9, false, false) == 9);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(9, false, true) == 10);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(-1, false, false) == -1);
    REQUIRE(vegas_hierarchical_valid_batch_inputs(1, true, true) == -1);

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

    const vegas_hierarchical_limits target_20 {
        /* .target_tokens   = */ 20,
        /* .max_tokens      = */ 20,
        /* .max_rounds      = */ 5,
        /* .max_corrections = */ 2,
    };
    REQUIRE(vegas_hierarchical_should_stop(target_20, 19, 4, 0, false, false, false) ==
            vegas_hierarchical_stop::continue_drafting);
    REQUIRE(vegas_hierarchical_should_stop(target_20, 20, 4, 0, false, false, false) ==
            vegas_hierarchical_stop::hard_cap);
    REQUIRE(vegas_hierarchical_should_stop(target_20, 16, 5, 0, false, false, false) ==
            vegas_hierarchical_stop::round_cap);
    REQUIRE(vegas_hierarchical_should_stop(target_20, 8, 2, 2, false, false, false) ==
            vegas_hierarchical_stop::correction_cap);

    return 0;
}
