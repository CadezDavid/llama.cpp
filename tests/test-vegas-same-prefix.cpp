#include "../examples/vegas/same-prefix.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, int line) {
    if (!condition) {
        std::fprintf(stderr, "requirement failed at line %d\n", line);
        std::abort();
    }
}

static bool near(double actual, double expected, double tolerance = 1e-12) {
    return std::abs(actual - expected) <= tolerance;
}

#define REQUIRE(condition) require((condition), __LINE__)

int main() {
    {
        const std::vector<float> dense  { 4.0f, 1.0f, -2.0f, 0.5f };
        const std::vector<float> sparse { 9.0f, 6.0f,  3.0f, 5.5f };

        const auto result = vegas_same_prefix_compare(dense, sparse, 3);

        REQUIRE(result.top1_match);
        REQUIRE(result.dense.top_token == 0);
        REQUIRE(result.sparse.top_token == 0);
        REQUIRE(result.dense_top_rank_in_sparse == 1);
        REQUIRE(result.sparse_top_rank_in_dense == 1);
        REQUIRE(near(result.kl_dense_sparse, 0.0));
        REQUIRE(near(result.kl_sparse_dense, 0.0));
        REQUIRE(near(result.jensen_shannon, 0.0));
        REQUIRE(near(result.total_variation, 0.0));
        REQUIRE(near(result.top_k_overlap, 1.0));
    }

    {
        const std::vector<float> dense  { 5.0f, 2.0f, 0.0f, -1.0f };
        const std::vector<float> sparse { 0.0f, 2.0f, 5.0f, -1.0f };

        const auto result = vegas_same_prefix_compare(dense, sparse, 2);

        REQUIRE(!result.top1_match);
        REQUIRE(result.dense.top_token == 0);
        REQUIRE(result.sparse.top_token == 2);
        REQUIRE(result.dense_top_rank_in_sparse == 3);
        REQUIRE(result.sparse_top_rank_in_dense == 3);
        REQUIRE(near(result.dense.top_margin, 3.0));
        REQUIRE(near(result.sparse.top_margin, 3.0));
        REQUIRE(result.kl_dense_sparse > 1.0);
        REQUIRE(result.kl_sparse_dense > 1.0);
        REQUIRE(result.jensen_shannon > 0.1);
        REQUIRE(result.total_variation > 0.5);
        REQUIRE(near(result.top_k_overlap, 0.5));
    }

    return 0;
}
