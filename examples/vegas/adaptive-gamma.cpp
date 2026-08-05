#include "adaptive-gamma.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <limits>

vegas_adaptive_gamma::vegas_adaptive_gamma(double beta) : beta_(beta) {
    if (!(beta_ >= 0.0 && beta_ < 1.0)) {
        beta_ = 0.9;
    }
    current_draft_.reserve(max_gamma);
}

void vegas_adaptive_gamma::begin_cycle() {
    current_draft_.clear();
    planned_gamma_ = cycles_ == 0 ? max_gamma : best_decision().gamma;
}

bool vegas_adaptive_gamma::observe_draft(const common_speculative_draft_observation & observation) {
    draft_record record;
    record.entropy = observation.entropy;
    record.top_probability = observation.top_probability;
    record.acceptance_probability = acceptance_probability(record.entropy, record.top_probability);
    current_draft_.push_back(record);

    if (cycles_ == 0) {
        planned_gamma_ = max_gamma;
        return observation.position < max_gamma;
    }

    planned_gamma_ = best_decision().gamma;
    return observation.position < planned_gamma_ && observation.position < max_gamma;
}

void vegas_adaptive_gamma::finish_cycle(
        int32_t drafted,
        int32_t accepted,
        int64_t draft_us,
        int64_t verify_us) {
    drafted = std::clamp(drafted, 0, max_gamma);
    accepted = std::clamp(accepted, 0, drafted);

    for (int32_t i = 0; i < accepted && i < (int32_t) current_draft_.size(); ++i) {
        update_ema(accepted_entropy_, current_draft_[i].entropy);
        update_ema(position_acceptance_[i + 1], 1.0);
    }
    if (accepted < drafted && accepted < (int32_t) current_draft_.size()) {
        update_ema(rejected_entropy_, current_draft_[accepted].entropy);
        update_ema(position_acceptance_[accepted + 1], 0.0);
    }

    if (drafted > 0) {
        update_ema(draft_costs_[drafted], (double) draft_us);
        update_ema(verify_costs_[drafted], (double) verify_us);
        gamma_histogram_[drafted]++;
    }

    cycles_++;
    drafted_total_ += drafted;
    accepted_total_ += accepted;
    draft_us_total_ += draft_us;
    verify_us_total_ += verify_us;
}

int32_t vegas_adaptive_gamma::planned_gamma() const {
    return planned_gamma_;
}

vegas_adaptive_gamma::decision vegas_adaptive_gamma::best_decision() const {
    decision best;
    best.efficiency = -std::numeric_limits<double>::infinity();

    for (int32_t gamma = min_gamma; gamma <= max_gamma; ++gamma) {
        const double value = efficiency(gamma);
        if (value > best.efficiency) {
            best.gamma = gamma;
            best.efficiency = value;
        }
    }
    return best;
}

std::string vegas_adaptive_gamma::summary_json() const {
    std::string histogram = "[";
    for (int32_t gamma = min_gamma; gamma <= max_gamma; ++gamma) {
        if (gamma > min_gamma) {
            histogram += ",";
        }
        histogram += std::to_string(gamma_histogram_[gamma]);
    }
    histogram += "]";

    const double mean_gamma = cycles_ > 0 ? (double) drafted_total_ / cycles_ : 0.0;
    const double accepted_entropy = accepted_entropy_.samples > 0 ? accepted_entropy_.value : 0.0;
    const double rejected_entropy = rejected_entropy_.samples > 0 ? rejected_entropy_.value : 0.0;

    char buffer[512];
    std::snprintf(buffer, sizeof(buffer),
            "\"adaptive_gamma\":true,\"adaptive_beta\":%.6f,"
            "\"adaptive_mean_gamma\":%.6f,\"adaptive_planned_gamma\":%d,"
            "\"adaptive_accepted_entropy\":%.6f,\"adaptive_rejected_entropy\":%.6f,"
            "\"adaptive_accepted_entropy_samples\":%" PRId64 ","
            "\"adaptive_rejected_entropy_samples\":%" PRId64 ","
            "\"adaptive_gamma_histogram\":%s",
            beta_, mean_gamma, planned_gamma_, accepted_entropy, rejected_entropy,
            accepted_entropy_.samples, rejected_entropy_.samples, histogram.c_str());
    return buffer;
}

bool vegas_adaptive_gamma::observer(
        void * userdata,
        const common_speculative_draft_observation & observation) {
    return static_cast<vegas_adaptive_gamma *>(userdata)->observe_draft(observation);
}

void vegas_adaptive_gamma::update_ema(ema_value & value, double sample) {
    if (value.samples == 0) {
        value.value = sample;
    } else {
        value.value = beta_ * value.value + (1.0 - beta_) * sample;
    }
    value.samples++;
}

double vegas_adaptive_gamma::acceptance_probability(double entropy, double top_probability) const {
    if (accepted_entropy_.samples > 0 && rejected_entropy_.samples > 0) {
        const double accepted = std::exp(-std::abs(entropy - accepted_entropy_.value));
        const double rejected = std::exp(-std::abs(entropy - rejected_entropy_.value));
        if (accepted + rejected > 0.0) {
            return accepted / (accepted + rejected);
        }
    }
    return std::clamp(top_probability, 0.01, 0.99);
}

double vegas_adaptive_gamma::prior_probability(int32_t position) const {
    if (position >= min_gamma && position <= max_gamma && position_acceptance_[position].samples > 0) {
        return std::clamp(position_acceptance_[position].value, 0.01, 0.99);
    }
    if (drafted_total_ > 0) {
        return std::clamp((double) accepted_total_ / drafted_total_, 0.01, 0.99);
    }
    return 0.5;
}

double vegas_adaptive_gamma::draft_cost(int32_t gamma) const {
    if (draft_costs_[gamma].samples > 0) {
        return draft_costs_[gamma].value;
    }
    if (drafted_total_ > 0 && draft_us_total_ > 0) {
        return (double) draft_us_total_ / drafted_total_ * gamma;
    }
    return (double) gamma;
}

double vegas_adaptive_gamma::verify_cost(int32_t gamma) const {
    if (verify_costs_[gamma].samples > 0) {
        return verify_costs_[gamma].value;
    }

    int32_t lower = 0;
    int32_t upper = 0;
    for (int32_t candidate = min_gamma; candidate <= max_gamma; ++candidate) {
        if (verify_costs_[candidate].samples == 0) {
            continue;
        }
        if (candidate <= gamma && (lower == 0 || candidate > lower)) {
            lower = candidate;
        }
        if (candidate >= gamma && (upper == 0 || candidate < upper)) {
            upper = candidate;
        }
    }

    if (lower > 0 && upper > 0 && lower != upper) {
        const double fraction = (double) (gamma - lower) / (upper - lower);
        return verify_costs_[lower].value + fraction * (verify_costs_[upper].value - verify_costs_[lower].value);
    }
    if (lower > 0) {
        return verify_costs_[lower].value;
    }
    if (upper > 0) {
        return verify_costs_[upper].value;
    }
    if (cycles_ > 0 && verify_us_total_ > 0) {
        return (double) verify_us_total_ / cycles_;
    }
    return 1.0;
}

double vegas_adaptive_gamma::efficiency(int32_t gamma) const {
    double expected_committed = 1.0;
    double prefix_probability = 1.0;
    for (int32_t position = min_gamma; position <= gamma; ++position) {
        const double probability = position <= (int32_t) current_draft_.size() ?
                current_draft_[position - 1].acceptance_probability : prior_probability(position);
        prefix_probability *= probability;
        expected_committed += prefix_probability;
    }

    const double cost = draft_cost(gamma) + verify_cost(gamma);
    return cost > 0.0 ? expected_committed / cost : 0.0;
}
