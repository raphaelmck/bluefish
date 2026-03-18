// solver.cpp - Shared solver infrastructure

#include "bluefish/solver.h"

#include <numeric>
#include<cassert>
#include <map>
#include <sstream>

namespace bluefish {

// - InfoNode -

void InfoNode::init(int n) {
	num_actions = n;
	regret_sum.assign(static_cast<std::size_t>(n), 0.0);
	strategy_sum.assign(static_cast<std::size_t>(n), 0.0);
}

std::vector<double> InfoNode::current_strategy() const {
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

std::vector<double> InfoNode::average_strategy() const {
	std::vector<double> avg(static_cast<std::size_t>(num_actions));
	double total = std::accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);

	if (total > 0.0) {
		for (int a = 0; a < num_actions; ++a) {
			avg[static_cast<std::size_t>(a)] = strategy_sum[static_cast<std::size_t>(a)] / total;
		}
	} else {
		std::fill(avg.begin(), avg.end(), 1.0 / static_cast<double>(num_actions));
	}
	return avg;
}

// - Exploitability - 

// Discover all info sets for br_player and their depth
void Solver::discover_info_sets(
	const GameState& state, int br_player, int br_depth,
	std::unordered_map<std::string, IsInfo>& out) const
{
	if (state.is_terminal()) return;

	if (state.is_chance()) {
		for (auto& outcome : state.chance_outcomes()) {
			discover_info_sets(*outcome.state, br_player, br_depth, out);
		}
		return;
	}
	
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
double Solver::accumulate_br(
	const GameState& state, int br_player,
	int target_depth, int br_depth, double opp_reach,
	const std::unordered_map<std::string, IsInfo>& info,
	const std::unordered_map<std::string, int>& resolved,
	std::unordered_map<std::string, std::vector<double>>& acc) const
{
	if (state.is_terminal()) {
		return opp_reach * state.utility(br_player);
	}

	if (state.is_chance()) {
		double total = 0.0;
		for (auto& outcome : state.chance_outcomes()) {
			total += accumulate_br(
				*outcome.state, br_player, target_depth, br_depth,
				opp_reach * outcome.probability, info, resolved, acc);
		}
		return total;
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
			auto next = state.act(actions[static_cast<std::size_t>(best_a)]);
			return accumulate_br(*next, br_player, target_depth, 
						br_depth + 1, opp_reach, info, resolved, acc);
		}

		if (depth == target_depth) {
			// This is the depth we're resolving: try all actions
			auto& vals = acc[key];
			double best_val = -std::numeric_limits<double>::infinity();
			for (int a = 0; a < n; ++a) {
				auto ai = static_cast<std::size_t>(a);
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
	} 
	// Opponent: follow their average strategy
	std::string key = state.info_set_key();
	auto it = nodes_.find(key);
	assert(it != nodes_.end());
	auto avg = it->second.average_strategy();

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

// Evaluate utility under a fully resolved BR strategy
double Solver::eval_br(
	const GameState& state, int br_player, double prob,
	const std::unordered_map<std::string, int>& br_actions) const
{
	if (state.is_terminal())
		return prob * state.utility(br_player);

	if (state.is_chance()) {
		double total = 0.0;
		for (auto& outcome : state.chance_outcomes()) {
			total += eval_br(*outcome.state, br_player,
							 prob * outcome.probability, br_actions);
		}
		return total;
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
		auto avg = it->second.average_strategy();

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
double Solver::exploitability(RootFn roots_fn) const {
	double total_exploit = 0.0;

	for (int br_player = 0; br_player < 2; ++br_player) {
		// Discover info sets and their depths
		std::unordered_map<std::string, IsInfo> info;
		{
			auto roots = roots_fn();
			for (auto& r : roots)
				discover_info_sets(*r.state, br_player, 0, info);
		}

		int max_depth = 0;
		for (auto& [k, v] : info) {
			max_depth = std::max(max_depth, v.depth);
		}

		// Resolve BR actions from deepest to shallowest
		std::unordered_map<std::string, int> br_actions;

		for (int d = max_depth; d >= 0; --d) {
			// Initialize accumulators for info sets at this depth
			std::unordered_map<std::string, std::vector<double>> acc;
			for (auto& [k, v] : info) {
				if (v.depth == d) {
					acc[k].assign(static_cast<std::size_t>(v.num_actions), 0.0);
				}
			}
			if (acc.empty()) continue;

			// Traverse all deals
			auto roots = roots_fn();
			for (auto& r : roots) {
				accumulate_br(*r.state, br_player, d, 0, r.probability, 
							  info, br_actions, acc);
			}

			for (auto& [k, vals] : acc) {
				int best_a = 0;
				for (int a = 1; a < static_cast<int>(vals.size()); ++a) {
					if (vals[static_cast<std::size_t>(a)] > 
						vals[static_cast<std::size_t>(best_a)]) {
							best_a = a;
					}
				}
				br_actions[k] = best_a;
			}
		}

		// Evaluate total BR value
		double br_value = 0.0;
		{
			auto roots = roots_fn();
			for (auto& r : roots) {
				br_value += eval_br(*r.state, br_player, r.probability, br_actions);
			}
		}
		total_exploit += br_value;
	}

	return total_exploit;
}

// - JSON serialization -

std::string Solver::serialize_json() const {
	std::map<std::string, const InfoNode*> sorted;
	for (auto& [key, node] : nodes_) 
		sorted[key] = &node;

	std::ostringstream ss;
	ss << "{\n";
	ss << "  \"algorithm\": \"" << name() << "\",\n";
	ss << "  \"iterations\": " << iterations_ << ",\n";
	ss << "  \"num_info_sets\": " << nodes_.size() << ",\n";
    ss << "  \"info_sets\": {\n";

	bool first = true;
	for (auto& [key, node] : sorted) {
		if (!first) ss << ",\n";
		first = false;
		auto avg = node->average_strategy();
		ss << "   \"" << key << "\" : [";
		for (int a = 0; a < node->num_actions; ++a) {
			if (a > 0) ss << ", ";
			ss << avg[static_cast<std::size_t>(a)];
		}
		ss << "]";
	}

	ss << "\n }\n}\n";
	ss << "}\n";
	return ss.str();
}

// - Validation -

std::string Solver::validate() const {
	for (auto& [key, node] : nodes_) {
		if (node.num_actions <= 0)
			return "info set '" + key + "' has num_actions=" + std::to_string(node.num_actions);

		auto n = static_cast<std::size_t>(node.num_actions);
		if (node.regret_sum.size() != n)
			return "info set '" + key + "' regret_sum size mismatch";
        if (node.strategy_sum.size() != n)
            return "info set '" + key + "' strategy_sum size mismatch";
 
        for (int a = 0; a < node.num_actions; ++a) {
            if (node.strategy_sum[static_cast<std::size_t>(a)] < 0.0)
                return "info set '" + key + "' has negative strategy_sum";
        }
    }
    return {};
}

} // namespace bluefish
