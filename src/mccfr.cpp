// mccfr.cpp - Exerternal-sampling Monte Carlo CFR

#include "bluefish/mccfr.h"
#include <random>

namespace bluefish {

MccfrSolver::MccfrSolver(uint64_t seed) : rng_(seed) {}

int MccfrSolver::sample_action(const std::vector<double>& probs) {
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double r = dist(rng_);
	double cum = 0.0;
	for (int a = 0; a < static_cast<int>(probs.size()); ++a) {
		cum += probs[static_cast<std::size_t>(a)];
		if (r < cum ) return a;
	}
	return static_cast<int>(probs.size()) - 1;
}

double MccfrSolver::train(int iterations, RootFn root_fn) {
	double total_value = 0.0;

	for (int t = 0; t < iterations; ++t) {
		int traverser = static_cast<int>(iterations_ % 2);

		auto roots = root_fn();
		std::vector<double> root_probs;
		root_probs.reserve(roots.size());
		for (auto& r: roots) 
			root_probs.push_back(r.probability);

		int root_idx = sample_action(root_probs);
		double v = traverse(*roots[static_cast<std::size_t>(root_idx)].state, traverser, 1.0, 1.0);

		if (traverser == 0)
			total_value += v;
		else
			total_value -= v;

		++iterations_;
	}

	return total_value / static_cast<double>(iterations);
}

double MccfrSolver::traverse(const GameState& state, int traverser,
							 double /*pi_traverser*/, double /*pi_opponent*/) {
	if (state.is_terminal())
		return state.utility(traverser);
	
	if (state.is_chance()) {
		auto outcomes = state.chance_outcomes();
		std::vector<double> probs;
		probs.reserve(outcomes.size());
		for (auto& o : outcomes) 
			probs.push_back(o.probability);
		int idx = sample_action(probs);
		return traverse(*outcomes[static_cast<std::size_t>(idx)].state, traverser, 1.0, 1.0);
	}

	int player = state.current_player();
	auto actions = state.legal_actions();
	auto n = static_cast<int>(actions.size());
	std::string key = state.info_set_key();

	auto& node = nodes_[key];
	if (node.num_actions == 0) node.init(n);
	auto sigma = node.current_strategy();

	if (player == traverser) {
		// - Traverser: explore all actions -
		std::vector<double> action_util(static_cast<std::size_t>(n));
		double node_util = 0.0;

		for (int a = 0; a < n; ++a) {
			auto ai = static_cast<std::size_t>(a);
			auto next = state.act(actions[ai]);
			action_util[ai] = traverse(*next, traverser, 1.0, 1.0);
			node_util += sigma[ai] * action_util[ai];
		}

		// Update regrets. no reach-probs weighting needed, the sampling handles it in expectation
		for (int a = 0; a < n; ++a) {
			auto ai = static_cast<std::size_t>(a);
			node.regret_sum[ai] += action_util[ai] - node_util;
		}

		return node_util;

	} else {
		// - Opponenet: sample one action -
		for (int a = 0; a < n; ++a) {
			node.strategy_sum[static_cast<std::size_t>(a)] += 
				sigma[static_cast<std::size_t>(a)];
		}

		int sampled = sample_action(sigma);
		auto next = state.act(actions[static_cast<std::size_t>(sampled)]);
		return traverse(*next, traverser, 1.0, 1.0);
	}
}

} // namespace bluefish
