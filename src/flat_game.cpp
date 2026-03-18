// flat_game.cpp - Game tree compilation

#include "bluefish/flat_game.h"

#include <cassert>

namespace bluefish {

void RegretTable::init(int num_info_sets, const std::vector<int>& is_na) {
	offset.resize(static_cast<std::size_t>(num_info_sets));
	num_actions.resize(static_cast<std::size_t>(num_info_sets));
	int total = 0;
	for (int i = 0; i < num_info_sets; ++i) {
		auto ui = static_cast<std::size_t>(i);
		offset[ui] = static_cast<int32_t>(total);
		num_actions[ui] = static_cast<int32_t>(is_na[ui]);
		total += is_na[ui];
	}
	total_entries = total;
	regret.assign(static_cast<std::size_t>(total), 0.0);
	strategy_sum.assign(static_cast<std::size_t>(total), 0.0);
}

void RegretTable::to_info_map(Solver::InfoMap& out, const std::vector<std::string>& is_keys) const {
	out.clear();
	auto nis = static_cast<int>(offset.size());
	for (int is = 0; is < nis; ++is) {
		auto uis = static_cast<std::size_t>(is);
		int na = num_actions[uis];
		int off = offset[uis];
		auto& node = out[is_keys[uis]];
		node.init(na);
		for (int a = 0; a < na; ++a ) {
			auto ua = static_cast<std::size_t>(a);
			auto uoa = static_cast<std::size_t>(off + a);
			node.regret_sum[ua] = regret[uoa];
			node.strategy_sum[ua] = strategy_sum[uoa];
		}
	}
}

// - FlatGame compilation -

namespace {

int build_subtree(FlatGame& g, const GameState& state, std::unordered_map<std::string, int>& key_to_id) {
	auto idx = static_cast<int>(g.nodes.size());
	g.nodes.push_back({});

	if (state.is_terminal()) {
		auto& n = g.nodes[static_cast<std::size_t>(idx)];
		n.type = FlatNode::kTerminal;
		n.player = 0;
		n.num_children = 0;
		n.info_set = -1;
		n.first_child = -1;
		n.utility_p0 = state.utility(0);
		return idx;
	}

	if (state.is_chance()) {
		auto outcomes = state.chance_outcomes();
		auto nc = static_cast<int>(outcomes.size());
		auto first = static_cast<int>(g.children.size());

		for (int a = 0; a < nc; ++a) {
			g.children.push_back(-1);
			g.chance_probs.push_back(outcomes[static_cast<std::size_t>(a)].probability);
		}

		// Fill node before recusion
		{
			auto& n = g.nodes[static_cast<std::size_t>(idx)];
			n.type = FlatNode::kChance;
			n.player = 0;
			n.num_children = static_cast<uint8_t>(nc);
			n.info_set = -1;
			n.first_child = first;
			n.utility_p0 = 0.0; 
		}

		for (int a = 0; a < nc; ++a){
			int child = build_subtree(
				g, *outcomes[static_cast<std::size_t>(a)].state, key_to_id);
			g.children[static_cast<std::size_t>(first + a)] = child;
		}
		return idx;
	}

	// Decision node
	auto actions = state.legal_actions();
	auto na = static_cast<int>(actions.size());
	assert(na <= kMaxActions);

	std::string key = state.info_set_key();
	auto [it, inserted] = key_to_id.try_emplace(
		key, static_cast<int>(key_to_id.size()));
	int is_id = it->second;

	auto first = static_cast<int>(g.children.size());
	for (int a = 0; a < na; ++a) {
		g.children.push_back(-1);
		g.chance_probs.push_back(0.0); // unused for decision nodes
	}

	{
		auto& n = g.nodes[static_cast<std::size_t>(idx)];
		n.type = FlatNode::kDecision;
		n.player = static_cast<uint8_t>(state.current_player());
		n.num_children = static_cast<uint8_t>(na);
		n.info_set = static_cast<int32_t>(is_id);
		n.first_child = first;
		n.utility_p0 = 0.0; 
	}

	for (int a = 0; a < na; ++a) {
		auto child_state = state.act(actions[static_cast<std::size_t>(a)]);
		int child = build_subtree(g, *child_state, key_to_id);
		g.children[static_cast<std::size_t>(first + a)] = child;
	}
	return idx;
}

} // namespace

FlatGame FlatGame::compile(Solver::RootFn root_fn) {
    FlatGame g;
    std::unordered_map<std::string, int> key_to_id;
 
    auto roots = root_fn();
    auto nr = static_cast<int>(roots.size());
 
    // Synthetic root chance node.
    auto root_idx = static_cast<int>(g.nodes.size());
    g.nodes.push_back({});
    auto first = static_cast<int>(g.children.size());
 
    for (int i = 0; i < nr; ++i) {
        g.children.push_back(-1);
        g.chance_probs.push_back(roots[static_cast<std::size_t>(i)].probability);
    }
 
    {
        auto& n = g.nodes[static_cast<std::size_t>(root_idx)];
        n.type = FlatNode::kChance;
        n.player = 0;
        n.num_children = static_cast<uint8_t>(nr);
        n.info_set = -1;
        n.first_child = first;
        n.utility_p0 = 0.0;
    }
 
    for (int i = 0; i < nr; ++i) {
        int child = build_subtree(
            g, *roots[static_cast<std::size_t>(i)].state, key_to_id);
        g.children[static_cast<std::size_t>(first + i)] = child;
    }
 
    // Build info set metadata.
    g.num_info_sets = static_cast<int>(key_to_id.size());
    g.is_num_actions.assign(static_cast<std::size_t>(g.num_info_sets), 0);
    g.is_keys.resize(static_cast<std::size_t>(g.num_info_sets));
 
    for (auto& [key, id] : key_to_id) {
        auto uid = static_cast<std::size_t>(id);
        g.is_keys[uid] = key;
    }
 
    // Derive num_actions per info set from the tree.
    for (auto& n : g.nodes) {
        if (n.type == FlatNode::kDecision) {
            auto uid = static_cast<std::size_t>(n.info_set);
            g.is_num_actions[uid] = n.num_children;
        }
    }
 
    return g;
}

int FlatGame::num_decision_nodes() const {
    int c = 0;
    for (auto& n : nodes) if (n.type == FlatNode::kDecision) ++c;
    return c;
}
 
int FlatGame::num_terminal_nodes() const {
    int c = 0;
    for (auto& n : nodes) if (n.type == FlatNode::kTerminal) ++c;
    return c;
}
 
int FlatGame::num_chance_nodes() const {
    int c = 0;
    for (auto& n : nodes) if (n.type == FlatNode::kChance) ++c;
    return c;
}

std::size_t FlatGame::memory_bytes() const {
    return nodes.size() * sizeof(FlatNode)
         + children.size() * sizeof(int32_t)
         + chance_probs.size() * sizeof(double)
         + is_keys.size() * sizeof(std::string); // approximate
}

} // namespace bluefish
