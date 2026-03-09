// cfr.cpp - Vanilla CFR + info-set-aware best response

#include "bluefish/cfr.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <cassert>

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

// Discover all info sets for br_player and their depth
void CfrTrainer::discover_info_sets(
	const GameState& state, int br_player, int br_depth,
	std::unordered_map<std::string, IsInfo>& out) const
{
	if (state.is_terminal()) return;
	
	auto actions = state.legal_actions();
	int player = state.current_player();

	if (player == br_player) {
		std::string key = state.info_set_key();
		if (out.find(key) == out.end()) {
			out[key] = {br_depth, static_cast<int>(actions.size())};
		}
		for (auto a : actions) {
			auto next = state.act(a);
			discover_info_sets(*next, br_player, br_depth + 1, out);
		}
	} else {
		for (auto a : actions) {
			auto next = state.act(a);
			discover_info_sets(*next, br_player, br_depth, out);
		}
	}
}

// Accumulate counterfactual action values for info sets at target_depth. At 
// deeper info sets, use already-resolved BR actions. At shallower info sets,
// traverse all actions 
double CfrTrainer::accumulate_br(
	const GameState& state, int br_player,
	int target_depth, int br_depth, double opp_reach,
	const std::unordered_map<std::string, IsInfo>& info,
	const std::unordered_map<std::string, int>& resolved,
	std::unordered_map<std::string, std::vector<double>>& acc) const
{
	if (state.is_terminal()) {
		return opp_reach * state.utility(br_player);
	}

	int player = state.current_player();
	auto actions = state.legal_actions();
	auto n = static_cast<int>(actions.size());

	if (player == br_player) {
		std::string key = state.info_set_key();
		int depth = info.at(key).depth;

		if (depth > target_depth) {
			// Already resolved, follow the computer BR action
			int best_a = resolved.at(key);
			auto next = state.act(actions[static_cast<std::atomic_size_t>(best_a)]);
			return accumulate_br(*next, br_player, target_depth, 
						br_depth + 1, opp_reach, info, resolved, acc);
		}

		if (depth == target_depth) {
			// This is the depth we're resolving: try all actions
			auto& vals = acc[key];
			double best_val = -std::numeric_limits<double>::infinity();
			for (int a = 0; a < n; ++a) {
				auto ai = static_cast<std::atomic_size_t>(a);
				auto next = state.act(actions[ai]);
				double v = accumulate_br(*next, br_player, target_depth, 
								br_depth + 1, opp_reach, info, resolved, acc);
				vals[ai] += v;
				best_val = std::max(best_val, v);
			}
			// Return optimistic value (exact choice made after aggregation)
			return best_val;
		}

		// depth < target.depth: shallower than target
		// Traverse all actions
		double total = 0.0;
		for (int a = 0; a < n; ++a) {
			auto ai = static_cast<std::size_t>(a);
			auto next = state.act(actions[ai]);
			total += accumulate_br(*next, br_player, target_depth,
								   br_depth + 1, opp_reach, info, resolved, acc);
		}
		return total;
	} else {
		// Opponent: follow their average strategy
		std::string key = state.info_set_key();
		auto it = nodes_.find(key);
		assert(it != nodes_.end());
		auto avg = it->second.average_srategy();

		double total = 0.0;
		for (int a = 0; a < n; ++a) {
			auto ai = static_cast<std::size_t>(a);
			auto next = state.act(actions[ai]);
			total += accumulate_br(*next, br_player, target_depth,
								   br_depth, opp_reach * avg[ai],
								   info, resolved, acc);
		}
		return total;
	}
}

// Evaluate utility under a fully resolved BR strategy
double CfrTrainer::eval_br(
		const GameState& state, int br_player, double prob,
		const std::unordered_map<std::string, int>& br_actions) const
{
	if (state.is_terminal()) {
		return prob * state.utility(br_player);
	}
	
	int player = state.current_player();
	auto actions = state.legal_actions();
	auto n = static_cast<int>(actions.size());

	if (player == br_player) {
		std::string key = state.info_set_key();
		int best_a = br_actions.at(key);
		auto next = state.act(actions[static_cast<std::size_t>(best_a)]);
		return eval_br(*next, br_player, prob, br_actions);
	} else {
		std::string key = state.info_set_key();
		auto it = nodes_.find(key);
		assert(it != nodes_.end());
		auto avg = it->second.average_srategy();

		double total = 0.0;
		for (int a = 0; a < n; ++a) {
			auto ai = static_cast<std::size_t>(a);
			auto next = state.act(actions[ai]);
			total += eval_br(*next, br_player, prob * avg[ai], br_actions);
		}
		return total;
	}
}

// Orchestrator: compute exploitability for both players
double CfrTrainer::exploitability(RootFn root_fr) const {
	double total_exploit = 0.0;

	for (int br_player = 0; br_player < 2; ++br_player) {
		// Discover info sets and their depths
	}
}

}
