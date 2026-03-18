// flat_game.cpp — Game tree compilation and flat evaluation.

#include "bluefish/flat_game.h"

#include <cassert>

namespace bluefish {

// - RegretTable -

void RegretTable::init(int num_info_sets,
                       const std::vector<int>& is_na) {
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

void RegretTable::avg_strategy(int is_id, double* out) const {
    auto uid = static_cast<std::size_t>(is_id);
    int off = offset[uid];
    int na = num_actions[uid];
    double total = 0.0;
    for (int a = 0; a < na; ++a)
        total += strategy_sum[static_cast<std::size_t>(off + a)];
    if (total > 0.0) {
        double inv = 1.0 / total;
        for (int a = 0; a < na; ++a)
            out[a] = strategy_sum[static_cast<std::size_t>(off + a)] * inv;
    } else {
        double u = 1.0 / static_cast<double>(na);
        for (int a = 0; a < na; ++a)
            out[a] = u;
    }
}

void RegretTable::to_info_map(
    Solver::InfoMap& out,
    const std::vector<std::string>& is_keys) const
{
    out.clear();
    auto nis = static_cast<int>(offset.size());
    for (int is = 0; is < nis; ++is) {
        auto uis = static_cast<std::size_t>(is);
        int na = num_actions[uis];
        int off = offset[uis];
        auto& node = out[is_keys[uis]];
        node.init(na);
        for (int a = 0; a < na; ++a) {
            auto ua = static_cast<std::size_t>(a);
            auto uoa = static_cast<std::size_t>(off + a);
            node.regret_sum[ua] = regret[uoa];
            node.strategy_sum[ua] = strategy_sum[uoa];
        }
    }
}

// - FlatGame compilation -

namespace {

int build_subtree(FlatGame& g, const GameState& state,
                  std::unordered_map<std::string, int>& key_to_id) {
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
            g.chance_probs.push_back(
                outcomes[static_cast<std::size_t>(a)].probability);
        }

        {
            auto& n = g.nodes[static_cast<std::size_t>(idx)];
            n.type = FlatNode::kChance;
            n.player = 0;
            n.num_children = static_cast<uint8_t>(nc);
            n.info_set = -1;
            n.first_child = first;
            n.utility_p0 = 0.0;
        }

        for (int a = 0; a < nc; ++a) {
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
        g.chance_probs.push_back(0.0);
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

void compute_depths(FlatGame& g, int ni, int p0_depth, int p1_depth) {
    auto& n = g.nodes[static_cast<std::size_t>(ni)];

    if (n.type == FlatNode::kTerminal) return;

    if (n.type == FlatNode::kChance) {
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a)
            compute_depths(g, g.children[static_cast<std::size_t>(fc + a)],
                           p0_depth, p1_depth);
        return;
    }

    // Decision node
    auto uid = static_cast<std::size_t>(n.info_set);
    int my_depth = (n.player == 0) ? p0_depth : p1_depth;

    if (g.is_depth[uid] < 0) {
        g.is_depth[uid] = my_depth;
        g.is_player[uid] = static_cast<int8_t>(n.player);
        g.max_depth[n.player] = std::max(g.max_depth[n.player], my_depth);
    }

    int next_p0 = p0_depth + ((n.player == 0) ? 1 : 0);
    int next_p1 = p1_depth + ((n.player == 1) ? 1 : 0);

    int fc = n.first_child;
    for (int a = 0; a < n.num_children; ++a)
        compute_depths(g, g.children[static_cast<std::size_t>(fc + a)],
                       next_p0, next_p1);
}

} // namespace

FlatGame FlatGame::compile(Solver::RootFn root_fn) {
    FlatGame g;
    std::unordered_map<std::string, int> key_to_id;

    auto roots = root_fn();
    auto nr = static_cast<int>(roots.size());

    // Synthetic root chance node
    auto root_idx = static_cast<int>(g.nodes.size());
    g.nodes.push_back({});
    auto first = static_cast<int>(g.children.size());

    for (int i = 0; i < nr; ++i) {
        g.children.push_back(-1);
        g.chance_probs.push_back(
            roots[static_cast<std::size_t>(i)].probability);
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

    // Build info set metadata
    g.num_info_sets = static_cast<int>(key_to_id.size());
    g.is_num_actions.assign(static_cast<std::size_t>(g.num_info_sets), 0);
    g.is_keys.resize(static_cast<std::size_t>(g.num_info_sets));
    g.is_player.resize(static_cast<std::size_t>(g.num_info_sets), -1);
    g.is_depth.assign(static_cast<std::size_t>(g.num_info_sets), -1);
    g.max_depth[0] = g.max_depth[1] = 0;

    for (auto& [key, id] : key_to_id) {
        auto uid = static_cast<std::size_t>(id);
        g.is_keys[uid] = key;
    }

    for (auto& n : g.nodes) {
        if (n.type == FlatNode::kDecision) {
            auto uid = static_cast<std::size_t>(n.info_set);
            g.is_num_actions[uid] = n.num_children;
        }
    }

    // Compute depths by DFS
    compute_depths(g, 0, 0, 0);

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
         + is_depth.size() * sizeof(int)
         + is_player.size() * sizeof(int8_t);
}

// Flat exploitability - best response on flat arrays

namespace {

double flat_accumulate_br(
    const FlatGame& game, const RegretTable& table,
    int ni, int br_player, int target_depth, int br_depth,
    double opp_reach,
    const std::vector<int>& resolved,  // resolved[is_id] = best action, -1 if unresolved
    std::vector<double>& acc)          // acc[offset + a] for info sets at target_depth
{
    auto& n = game.nodes[static_cast<std::size_t>(ni)];

    if (n.type == FlatNode::kTerminal) {
        double u = (br_player == 0) ? n.utility_p0 : -n.utility_p0;
        return opp_reach * u;
    }

    if (n.type == FlatNode::kChance) {
        double total = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            total += flat_accumulate_br(
                game, table, game.children[ua], br_player,
                target_depth, br_depth,
                opp_reach * game.chance_probs[ua], resolved, acc);
        }
        return total;
    }

    // Decision node
    int is = n.info_set;
    int na = n.num_children;
    int fc = n.first_child;

    if (n.player == br_player) {
        int depth = game.is_depth[static_cast<std::size_t>(is)];

        if (depth > target_depth) {
            // Already resolved - follow best action
            int best_a = resolved[static_cast<std::size_t>(is)];
            return flat_accumulate_br(
                game, table, game.children[static_cast<std::size_t>(fc + best_a)],
                br_player, target_depth, br_depth + 1,
                opp_reach, resolved, acc);
        }

        if (depth == target_depth) {
            // Accumulate action values
            int off = table.offset[static_cast<std::size_t>(is)];
            double best_val = -std::numeric_limits<double>::infinity();
            for (int a = 0; a < na; ++a) {
                double v = flat_accumulate_br(
                    game, table, game.children[static_cast<std::size_t>(fc + a)],
                    br_player, target_depth, br_depth + 1,
                    opp_reach, resolved, acc);
                acc[static_cast<std::size_t>(off + a)] += v;
                best_val = std::max(best_val, v);
            }
            return best_val;
        }

        // Shallower - traverse all actions (counterfactual)
        double total = 0.0;
        for (int a = 0; a < na; ++a) {
            total += flat_accumulate_br(
                game, table, game.children[static_cast<std::size_t>(fc + a)],
                br_player, target_depth, br_depth + 1,
                opp_reach, resolved, acc);
        }
        return total;
    }

    // Opponent - follow average strategy
    double avg[kMaxActions];
    table.avg_strategy(is, avg);

    double total = 0.0;
    for (int a = 0; a < na; ++a) {
        total += flat_accumulate_br(
            game, table, game.children[static_cast<std::size_t>(fc + a)],
            br_player, target_depth, br_depth,
            opp_reach * avg[a], resolved, acc);
    }
    return total;
}

// Evaluate total BR value with all actions resolved
double flat_eval_br(
    const FlatGame& game, const RegretTable& table,
    int ni, int br_player, double prob,
    const std::vector<int>& br_actions)
{
    auto& n = game.nodes[static_cast<std::size_t>(ni)];

    if (n.type == FlatNode::kTerminal) {
        double u = (br_player == 0) ? n.utility_p0 : -n.utility_p0;
        return prob * u;
    }

    if (n.type == FlatNode::kChance) {
        double total = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            total += flat_eval_br(
                game, table, game.children[ua], br_player,
                prob * game.chance_probs[ua], br_actions);
        }
        return total;
    }

    int is = n.info_set;
    int na = n.num_children;
    int fc = n.first_child;

    if (n.player == br_player) {
        int best_a = br_actions[static_cast<std::size_t>(is)];
        return flat_eval_br(
            game, table,
            game.children[static_cast<std::size_t>(fc + best_a)],
            br_player, prob, br_actions);
    }

    // Opponent - average strategy
    double avg[kMaxActions];
    table.avg_strategy(is, avg);

    double total = 0.0;
    for (int a = 0; a < na; ++a) {
        total += flat_eval_br(
            game, table, game.children[static_cast<std::size_t>(fc + a)],
            br_player, prob * avg[a], br_actions);
    }
    return total;
}

} // namespace

double flat_exploitability(const FlatGame& game, const RegretTable& table) {
    double total = 0.0;

    for (int br_player = 0; br_player < 2; ++br_player) {
        int md = game.max_depth[br_player];

        // resolved[is_id] = best action index, -1 if not yet resolved
        std::vector<int> resolved(static_cast<std::size_t>(game.num_info_sets), -1);

        for (int d = md; d >= 0; --d) {
            // Initialize accumulator for info sets at this depth
            // Reuse the same layout as RegretTable for easy indexing
            std::vector<double> acc(static_cast<std::size_t>(table.total_entries), 0.0);

            // Check if any info sets at this depth belong to br_player
            bool any = false;
            for (int is = 0; is < game.num_info_sets; ++is) {
                auto uis = static_cast<std::size_t>(is);
                if (game.is_player[uis] == br_player && game.is_depth[uis] == d)
                    any = true;
            }
            if (!any) continue;

            // Single traversal to accumulate action values
            flat_accumulate_br(game, table, 0, br_player, d, 0,
                               1.0, resolved, acc);

            // Pick best action at each info set at this depth
            for (int is = 0; is < game.num_info_sets; ++is) {
                auto uis = static_cast<std::size_t>(is);
                if (game.is_player[uis] != br_player ||
                    game.is_depth[uis] != d) continue;

                int off = table.offset[uis];
                int na = table.num_actions[uis];
                int best_a = 0;
                for (int a = 1; a < na; ++a) {
                    if (acc[static_cast<std::size_t>(off + a)] >
                        acc[static_cast<std::size_t>(off + best_a)])
                        best_a = a;
                }
                resolved[uis] = best_a;
            }
        }

        // Final evaluation with all actions resolved
        total += flat_eval_br(game, table, 0, br_player, 1.0, resolved);
    }

    return total;
}

} // namespace bluefish
