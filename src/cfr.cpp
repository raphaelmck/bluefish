// cfr.cpp - Vanilla CFR + info-set-aware best response

#include "bluefish/cfr.h"

namespace bluefish {

void InfoNode::init(int n) {
	num_actions = n;
	regret_sum.assign(static_cast<std::size_t>(n), 0.0);
	strategy_sum.assign(static_cast<std::size_t>(n), 0.0);
}

std::vector<double> InfoNode::current_stategy() const {
	std::vector<double> strat(static_cast<std::size_t>(num_actions));
	double pos_sum = 0.0;

	for (int a = 0; a < num_actions; ++a) {
		auto ai = static_cast<std::size_t>(a);
		strat[ai] = std::max(0.0, regret_sum[ai]);
		pos_sum += strat[ai];
	}

	if (pos_sum > 0.0) {
		for (auto& s : strat) s /= pos_sum;
	} else {
		std::fill(strat.begin(), strat.end(), 1.0 / static_cast<double>(num_actions));
	}
	return strat;
}

double CfrTrainer::train(int iterations, RootFn root_fn) {
	double total_value = 0.0;

	for (int t = 0; t < iterations; ++t) {
		auto roots = root_fn();
		double iter_value = 0.0;

		for (auto& root : roots) {
			double cp = root.probability;
			iter_value += cp * cfr(*root.state, cp, cp);
		}

		total_value += iter_value;
		++iterations_;
	}

	return total_value / static_cast<double>(iterations);
}
}
