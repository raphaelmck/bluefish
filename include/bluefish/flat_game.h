// bluefish/flat_game.h - Compiled game tree for high-performance traversal

#pragma once

#include "solver.h"

namespace bluefish {

inline constexpr int kMaxActions = 4;

struct FlatNode {
	enum Type : uint8_t { kDecision = 0, kChance = 1, kTerminal = 2 } ;
	Type type;
	uint8_t player;
	uint8_t num_children; // 0 or 1 (decision nodes only)
	uint8_t _pad = 0;
	int32_t info_set; // dense ID, -1 for non-deicision
	int32_t first_child; // index into children[]
	double utility_p0;
};

// - Regret table: flat arrays indexed by info set ID -

struct RegretTable {
	std::vector<double> regret;
	std::vector<double> strategy_sum;
	std::vector<int32_t> offset;
	std::vector<int32_t> num_actions;
	int total_entries = 0;

	void init(int num_info_sets, const std::vector<int>& is_num_actions);

	void avg_strategy(int is_id, double* out) const;

	void to_info_map(Solver::InfoMap& out, const std::vector<std::string>& is_keys) const;
};

// - Compiled game tree -

struct FlatGame {
	std::vector<FlatNode> nodes;
	std::vector<int32_t> children;
	std::vector<double> chance_probs;

	int num_info_sets = 0;
	std::vector<int> is_num_actions;
	std::vector<std::string> is_keys;

	std::vector<int8_t> is_player;
	std::vector<int> is_depth;
	int max_depth[2] = {0, 0};

	static FlatGame compile(Solver::RootFn root_fn);

	int num_nodes() const { return static_cast<int>(nodes.size()); }
	int num_decision_nodes() const;
	int num_terminal_nodes() const;
	int num_chance_nodes() const;

	std::size_t memory_bytes() const;
};

// - Flat exploitability -

double flat_exploitability(const FlatGame& game, const RegretTable& table);

} // namespace bluefish
