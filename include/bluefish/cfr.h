// bluefish/cfr.h - Vanilla Counterfactual Regeret Minimization

#pragma once

#include "game.h"

#include <functional>
#include <unordered_map>

namespace bluefish {

// Information set node
struct InfoNode {
	std::vector<double> regret_sum{};
	std::vector<double> strategy_sum{};
	int num_actions = 0;

	void init(int n);
	std::vector<double> current_strategy() const;
	std::vector<double> average_strategy() const;
};

// CFR trainer
class CfrTrainer {
public:
	using RootFn = std::function<std::vector<RootState>()>;
	using InfoMap = std::unordered_map<std::string, InfoNode>;

	CfrTrainer() = default;

	double train(int iterations, RootFn root_fn);

	double exploitability(RootFn root_fr) const;

	const InfoMap& info_map() const { return nodes_; }
	std::size_t num_info_sets() const { return nodes_.size(); }
	int64_t iterations() const { return iterations_; }

private:
	InfoMap nodes_{};
	int64_t iterations_ = 0;

	// Recursive CFR. Returns utility for player 0
	double cfr(const GameState& state, double pi0, double pi);

	// Best response (info-set-aware)
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
