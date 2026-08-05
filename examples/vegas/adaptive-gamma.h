#pragma once

#include "speculative.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class vegas_adaptive_gamma {
public:
    static constexpr int32_t min_gamma = 1;
    static constexpr int32_t max_gamma = 10;

    struct decision {
        int32_t gamma = min_gamma;
        double efficiency = 0.0;
    };

    explicit vegas_adaptive_gamma(double beta = 0.9);

    void begin_cycle();
    bool observe_draft(const common_speculative_draft_observation & observation);
    void finish_cycle(int32_t drafted, int32_t accepted, int64_t draft_us, int64_t verify_us);

    int32_t planned_gamma() const;
    decision best_decision() const;
    std::string summary_json() const;

    static bool observer(void * userdata, const common_speculative_draft_observation & observation);

private:
    struct ema_value {
        double value = 0.0;
        int64_t samples = 0;
    };

    struct draft_record {
        double entropy = 0.0;
        double top_probability = 0.0;
        double acceptance_probability = 0.0;
    };

    void update_ema(ema_value & value, double sample);
    double acceptance_probability(double entropy, double top_probability) const;
    double prior_probability(int32_t position) const;
    double draft_cost(int32_t gamma) const;
    double verify_cost(int32_t gamma) const;
    double efficiency(int32_t gamma) const;

    double beta_;
    int64_t cycles_ = 0;
    int64_t drafted_total_ = 0;
    int64_t accepted_total_ = 0;
    int64_t draft_us_total_ = 0;
    int64_t verify_us_total_ = 0;
    int32_t planned_gamma_ = max_gamma;

    ema_value accepted_entropy_;
    ema_value rejected_entropy_;
    std::array<ema_value, max_gamma + 1> position_acceptance_ {};
    std::array<ema_value, max_gamma + 1> draft_costs_ {};
    std::array<ema_value, max_gamma + 1> verify_costs_ {};
    std::array<int64_t, max_gamma + 1> gamma_histogram_ {};
    std::vector<draft_record> current_draft_;
};
