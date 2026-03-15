// cfr_plus.cpp

#include "bluefish/cfr_plus.h"

namespace bluefish {

double CfrPlusSolver::train(int iterations, RootFn root_fn) {
	double total_value = 0.0;
	for (int t = 0; t < iterations; ++t) {
		auto roots = root_fn();
		double iter_value = 0.0;
		for (auto& root : roots) {
			double cp = root.probability;
			iter_value += cp * cfr_plus(*root.state, cp, cp);
		}
		total_value += iter_value;
		++iterations_;
	}
	return total_value / static_cast<double>(iterations);
}

double CfrPlusSolver::cfr_plus(const GameState& state, double pi0, double pi1) {
	++stats_.nodes_touched;

	if (state.is_terminal()) {
		++stats_.terminal_visits;
		return state.utility(0);
	}

	if (state.is_chance()) {
		++stats_.chance_visits;
		double ev = 0.0;
		for (auto& o : state.chance_outcomes())
			ev += o.probability * cfr_plus(*o.state, pi0 * o.probability, 
										pi1 * o.probability);
		return ev;
	}

	++stats_.info_set_visits;

	int player = state.current_player();
	auto actions = state.legal_actions();
	auto n = static_cast<int>(actions.size());
	std::string key = state.info_set_key();

	auto& node = nodes_[key];
	if (node.num_actions == 0) node.init(n);
	auto sigma = node.current_strategy();

	std::vector<double> action_util(static_cast<std::size_t>(n));
	double node_util = 0.0;

	for (int a = 0; a < n; ++a) {
		auto ai = static_cast<std::size_t>(a);
		auto next = state.act(actions[ai]);
		if (player == 0)
			action_util[ai] = cfr_plus(*next, pi0 * sigma[ai], pi1);
		else
			action_util[ai] = cfr_plus(*next, pi0, pi1 * sigma[ai]);

		node_util += sigma[ai] * action_util[ai];
	}

	double t_weight = static_cast<double>(iterations_ + 1);

	for (int a = 0; a < n; ++a) {
		auto ai = static_cast<std::size_t>(a);
		if (player == 0) {
			node.regret_sum[ai] += pi1 * (action_util[ai] - node_util);
			node.strategy_sum[ai] += t_weight * pi0 * sigma[ai];
		} else {
			node.regret_sum[ai] += pi0 * (node_util - action_util[ai]);
			node.strategy_sum[ai] += t_weight * pi1 * sigma[ai];
		}
		// CFR+ floor regrets at zero inline
		node.regret_sum[ai] = std::max(0.0, node.regret_sum[ai]);
	}

	return node_util;
}

std::string CfrPlusSolver::validate() const {
	std::string base_err = Solver::validate();
	if (!base_err.empty()) return base_err;
 
    // CFR+-specific: all cumulative regrets must be >= 0.
    for (auto& [key, node] : nodes_) {
        for (int a = 0; a < node.num_actions; ++a) {
            if (node.regret_sum[static_cast<std::size_t>(a)] < -1e-12)
                return "cfr+ invariant: info set '" + key +
                       "' has negative regret " +
                       std::to_string(node.regret_sum[static_cast<std::size_t>(a)]);
        }
    }
    return {};
}

} // namespace bluefish
