// bluefish/solver.h - Shared solver infrastructure

#pragma once

#include "game.h"

#include <functional>

namespace bluefish {

// - Information set node -
// Stores accumulated regrets and strategy sums for one information set
// All CFR-family algorithms use the same node structure and differ on how they 
// update the regret and strategy sum

struct InfoNode {
	std::vector<double> regret_sum;
	std::vector<double> strategy_sum;
	int num_actions = 0;

	void init(int n);

	// Regret matching, uniform fallback when all regrets <= 0
	std::vector<double> current_strategy() const;
	// Time-averaged strategy - the convergent Nash output
	std::vector<double> average_strategy() const;
};

// - Traversal statistics -
// Lightweight counters incremented during CFR traversal
// All counters are cumulative accross train() calls

struct Stats {
	int64_t nodes_touched = 0;
	int64_t terminal_visits = 0;
	int64_t info_set_visits = 0;
	int64_t chance_visits = 0;

	void reset() {
		nodes_touched = terminal_visits = info_set_visits = chance_visits = 0;
	}
};

// - Solver base class -

class Solver {
public:
	using RootFn = std::function<std::vector<RootState>()>;
	using InfoMap = std::unordered_map<std::string, InfoNode>;

	virtual ~Solver() = default;

	virtual double train(int iterations, RootFn root_rn) = 0;

	virtual std::string name() const = 0;

	// - Shared evaluation tools -

	double exploitability(RootFn root_fn) const;
	std::string serialize_json() const;

	// - Validation -

	virtual std::string validate() const;

	// - Accessors -

	const InfoMap& info_map() const { return nodes_; }
	std::size_t num_info_sets() const { return nodes_.size(); }
	int64_t iterations() const { return iterations_; }
	const Stats& stats() const { return stats_; }

protected:
	InfoMap nodes_;
	int64_t iterations_ = 0;
	Stats stats_;

private:
	// - Best response internals -
	struct IsInfo { int depth; int num_actions; };

	void discover_info_sets(
		const GameState& state, int br_player, int br_depth,
		std::unordered_map<std::string, IsInfo>& out) const;
	
	double accumulate_br(
		const GameState& state, int br_player,
		int target_depth, int br_depth, double opp_reach,
		const std::unordered_map<std::string, IsInfo>& info,
		const std::unordered_map<std::string, int>& resolved,
		std::unordered_map<std::string, std::vector<double>>& acc) const;
	
	double eval_br(
		const GameState& state, int br_player, double prob,
		const std::unordered_map<std::string, int>& br_actions) const;
};

} // namespace bluefish
