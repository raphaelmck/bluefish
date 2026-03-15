// cfr.cpp - Vanilla CFR + info-set-aware best response

#include "bluefish/cfr.h"

namespace bluefish {

double CfrSolver::train(int iterations, RootFn root_fn) {
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

double CfrSolver::cfr(const GameState& state, double pi0, double pi1) {
	++stats_.nodes_touched;

	if (state.is_terminal()) {
		++stats_.terminal_visits;
		return state.utility(0);
	}

	if (state.is_chance()) {
		++stats_.chance_visits;
		double ev = 0.0;
		for (auto& o : state.chance_outcomes())
			ev += o.probability * cfr(*o.state, pi0 * o.probability,
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
			action_util[ai] = cfr(*next, pi0 * sigma[ai], pi1);
		else
			action_util[ai] = cfr(*next, pi0, pi1 * sigma[ai]);
		node_util += sigma[ai] * action_util[ai];
	}

	for (int a = 0; a < n; ++a) {
		auto ai = static_cast<std::size_t>(a);
		if (player == 0) {
			node.regret_sum[ai] += pi1 * (action_util[ai] - node_util);
			node.strategy_sum[ai] += pi0 * sigma[ai];
		} else {
			node.regret_sum[ai] += pi0 * (node_util - action_util[ai]);
			node.strategy_sum[ai] += pi1 * sigma[ai];
		}
	}
	return node_util;
}

} // namespace bluefish
