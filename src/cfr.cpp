// cfr.cpp - Vanilla CFR + info-set-aware best response

#include "bluefish/cfr.h"
#include <unordered_map>

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

double CfrTrainer::cfr(const GameState& state, double pi0, double pi1) {
	if (state.is_terminal()) {
		return state.utility(0);
	}

	int player = state.current_player();
	auto actions = state.legal_actions();
	auto n = static_cast<int>(actions.size());
	std::string key = state.info_set_key();

	auto node = nodes_[key];
	if (node.num_actions == 0) {
		node.init(n);
	}

	auto sigma = node.current_stategy();

	std::vector<double> action_util(static_cast<std::size_t>(n));
	double node_util = 0.0;

	for (int a = 0; a < n; ++a) {
		auto ai = static_cast<std::size_t>(a);
		auto next = state.act(actions[ai]);
		
		if (player == 0) {
			action_util[ai] = cfr(*next, pi0 * sigma[ai], pi1);
		} else {
			action_util[ai] = cfr(*next, pi0, pi1 * sigma[ai]);
		}
		node_util += sigma[ai] * action_util[ai];
	}

	// Counterfactual regret updates
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

// Exploitability
void CfrTrainer::discover_info_sets(
	const GameState& state, int br_player, int br_depth,
	std::unordered_map<std::string, IsInfo>& out) const
{
	if (state.is_terminal()) return;
	
}


}
