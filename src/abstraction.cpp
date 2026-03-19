// abstraction.cpp — Card abstraction layer.

#include "bluefish/abstraction.h"

#include <cassert>

namespace bluefish {

// - make_abstraction - 

Abstraction make_abstraction(
    const FlatGame& exact_game,
    const std::string& abs_name,
    std::function<std::string(const std::string&)> remap)
{
    Abstraction abs;
    abs.name = abs_name;
    abs.mapping.resize(static_cast<std::size_t>(exact_game.num_info_sets));

    std::unordered_map<std::string, int> key_to_id;

    for (int i = 0; i < exact_game.num_info_sets; ++i) {
        auto ui = static_cast<std::size_t>(i);
        std::string akey = remap(exact_game.is_keys[ui]);
        auto [it, inserted] = key_to_id.try_emplace(
            akey, static_cast<int>(key_to_id.size()));
        abs.mapping[ui] = it->second;
    }

    abs.num_abstract = static_cast<int>(key_to_id.size());
    abs.abstract_keys.resize(static_cast<std::size_t>(abs.num_abstract));
    for (auto& [key, id] : key_to_id)
        abs.abstract_keys[static_cast<std::size_t>(id)] = key;

    return abs;
}

// - apply_abstraction -

FlatGame apply_abstraction(const FlatGame& exact, const Abstraction& abs) {
    FlatGame g = exact;

    for (auto& n : g.nodes) {
        if (n.type == FlatNode::kDecision)
            n.info_set = static_cast<int32_t>(
                abs.mapping[static_cast<std::size_t>(n.info_set)]);
    }

    g.num_info_sets = abs.num_abstract;
    g.is_keys = abs.abstract_keys;
    g.is_num_actions.assign(static_cast<std::size_t>(abs.num_abstract), 0);
    g.is_player.assign(static_cast<std::size_t>(abs.num_abstract), -1);
    g.is_depth.assign(static_cast<std::size_t>(abs.num_abstract), -1);
    g.max_depth[0] = g.max_depth[1] = 0;

    for (auto& n : g.nodes) {
        if (n.type == FlatNode::kDecision) {
            auto uid = static_cast<std::size_t>(n.info_set);
            if (g.is_num_actions[uid] == 0)
                g.is_num_actions[uid] = n.num_children;
            assert(g.is_num_actions[uid] == n.num_children);
        }
    }

    // Recompute depths
    struct Dfs {
        static void run(FlatGame& g, int ni, int p0d, int p1d) {
            auto& n = g.nodes[static_cast<std::size_t>(ni)];
            if (n.type == FlatNode::kTerminal) return;
            if (n.type == FlatNode::kChance) {
                int fc = n.first_child;
                for (int a = 0; a < n.num_children; ++a)
                    run(g, g.children[static_cast<std::size_t>(fc + a)],
                        p0d, p1d);
                return;
            }
            auto uid = static_cast<std::size_t>(n.info_set);
            int my_depth = (n.player == 0) ? p0d : p1d;
            if (g.is_depth[uid] < 0 || my_depth < g.is_depth[uid]) {
                g.is_depth[uid] = my_depth;
                g.is_player[uid] = static_cast<int8_t>(n.player);
            }
            g.max_depth[n.player] = std::max(g.max_depth[n.player], my_depth);
            int fc = n.first_child;
            for (int a = 0; a < n.num_children; ++a)
                run(g, g.children[static_cast<std::size_t>(fc + a)],
                    p0d + ((n.player == 0) ? 1 : 0),
                    p1d + ((n.player == 1) ? 1 : 0));
        }
    };
    Dfs::run(g, 0, 0, 0);

    return g;
}

// - lifted_flat_exploitability -

namespace {

struct OffsetLayout {
    std::vector<int32_t> off;
    int total = 0;

    explicit OffsetLayout(const FlatGame& g) {
        off.resize(static_cast<std::size_t>(g.num_info_sets));
        int t = 0;
        for (int i = 0; i < g.num_info_sets; ++i) {
            off[static_cast<std::size_t>(i)] = static_cast<int32_t>(t);
            t += g.is_num_actions[static_cast<std::size_t>(i)];
        }
        total = t;
    }
};

double lifted_acc(
    const FlatGame& ex, const RegretTable& atab, const Abstraction& abs,
    const OffsetLayout& lay,
    int ni, int brp, int td, int bd, double opp_reach,
    const std::vector<int>& resolved, std::vector<double>& acc)
{
    auto& n = ex.nodes[static_cast<std::size_t>(ni)];

    if (n.type == FlatNode::kTerminal) {
        double u = (brp == 0) ? n.utility_p0 : -n.utility_p0;
        return opp_reach * u;
    }
    if (n.type == FlatNode::kChance) {
        double tot = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            tot += lifted_acc(ex, atab, abs, lay,
                              ex.children[ua], brp, td, bd,
                              opp_reach * ex.chance_probs[ua],
                              resolved, acc);
        }
        return tot;
    }

    int is = n.info_set;
    int na = n.num_children;
    int fc = n.first_child;

    if (n.player == brp) {
        int depth = ex.is_depth[static_cast<std::size_t>(is)];

        if (depth > td) {
            int best_a = resolved[static_cast<std::size_t>(is)];
            return lifted_acc(ex, atab, abs, lay,
                              ex.children[static_cast<std::size_t>(fc + best_a)],
                              brp, td, bd + 1, opp_reach, resolved, acc);
        }
        if (depth == td) {
            int off = lay.off[static_cast<std::size_t>(is)];
            double best_val = -std::numeric_limits<double>::infinity();
            for (int a = 0; a < na; ++a) {
                double v = lifted_acc(ex, atab, abs, lay,
                    ex.children[static_cast<std::size_t>(fc + a)],
                    brp, td, bd + 1, opp_reach, resolved, acc);
                acc[static_cast<std::size_t>(off + a)] += v;
                best_val = std::max(best_val, v);
            }
            return best_val;
        }
        double tot = 0.0;
        for (int a = 0; a < na; ++a) {
            tot += lifted_acc(ex, atab, abs, lay,
                ex.children[static_cast<std::size_t>(fc + a)],
                brp, td, bd + 1, opp_reach, resolved, acc);
        }
        return tot;
    }

    // Opponent: look up abstract strategy
    int abstract_is = abs.mapping[static_cast<std::size_t>(is)];
    double avg[kMaxActions];
    atab.avg_strategy(abstract_is, avg);

    double tot = 0.0;
    for (int a = 0; a < na; ++a) {
        tot += lifted_acc(ex, atab, abs, lay,
            ex.children[static_cast<std::size_t>(fc + a)],
            brp, td, bd, opp_reach * avg[a], resolved, acc);
    }
    return tot;
}

double lifted_eval(
    const FlatGame& ex, const RegretTable& atab, const Abstraction& abs,
    int ni, int brp, double prob,
    const std::vector<int>& br_actions)
{
    auto& n = ex.nodes[static_cast<std::size_t>(ni)];

    if (n.type == FlatNode::kTerminal) {
        double u = (brp == 0) ? n.utility_p0 : -n.utility_p0;
        return prob * u;
    }
    if (n.type == FlatNode::kChance) {
        double tot = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            tot += lifted_eval(ex, atab, abs, ex.children[ua],
                               brp, prob * ex.chance_probs[ua], br_actions);
        }
        return tot;
    }

    int is = n.info_set;
    int na = n.num_children;
    int fc = n.first_child;

    if (n.player == brp) {
        int best_a = br_actions[static_cast<std::size_t>(is)];
        return lifted_eval(ex, atab, abs,
            ex.children[static_cast<std::size_t>(fc + best_a)],
            brp, prob, br_actions);
    }

    int abstract_is = abs.mapping[static_cast<std::size_t>(is)];
    double avg[kMaxActions];
    atab.avg_strategy(abstract_is, avg);

    double tot = 0.0;
    for (int a = 0; a < na; ++a) {
        tot += lifted_eval(ex, atab, abs,
            ex.children[static_cast<std::size_t>(fc + a)],
            brp, prob * avg[a], br_actions);
    }
    return tot;
}

} // namespace

double lifted_flat_exploitability(
    const FlatGame& exact_game,
    const RegretTable& abstract_table,
    const Abstraction& abs)
{
    OffsetLayout layout(exact_game);
    double total = 0.0;

    for (int brp = 0; brp < 2; ++brp) {
        int md = exact_game.max_depth[brp];
        std::vector<int> resolved(
            static_cast<std::size_t>(exact_game.num_info_sets), -1);

        for (int d = md; d >= 0; --d) {
            std::vector<double> acc(
                static_cast<std::size_t>(layout.total), 0.0);

            bool any = false;
            for (int is = 0; is < exact_game.num_info_sets; ++is) {
                auto uis = static_cast<std::size_t>(is);
                if (exact_game.is_player[uis] == brp &&
                    exact_game.is_depth[uis] == d)
                    any = true;
            }
            if (!any) continue;

            lifted_acc(exact_game, abstract_table, abs, layout,
                       0, brp, d, 0, 1.0, resolved, acc);

            for (int is = 0; is < exact_game.num_info_sets; ++is) {
                auto uis = static_cast<std::size_t>(is);
                if (exact_game.is_player[uis] != brp ||
                    exact_game.is_depth[uis] != d) continue;
                int off = layout.off[uis];
                int na = exact_game.is_num_actions[uis];
                int best_a = 0;
                for (int a = 1; a < na; ++a)
                    if (acc[static_cast<std::size_t>(off + a)] >
                        acc[static_cast<std::size_t>(off + best_a)])
                        best_a = a;
                resolved[uis] = best_a;
            }
        }

        total += lifted_eval(exact_game, abstract_table, abs,
                             0, brp, 1.0, resolved);
    }
    return total;
}

// - Leduc abstractions -

namespace leduc_abs {

Abstraction exact(const FlatGame& game) {
    return make_abstraction(game, "exact",
        [](const std::string& key) { return key; });
}

Abstraction jq_merge(const FlatGame& game) {
    return make_abstraction(game, "jq-merge",
        [](const std::string& key) -> std::string {
            std::string out = key;
            auto colon = out.find(':');
            if (colon == std::string::npos) return out;
            for (std::size_t i = 0; i < colon; ++i) {
                if (out[i] == 'J' || out[i] == 'Q') out[i] = 'L';
                else if (out[i] == 'K') out[i] = 'H';
            }
            return out;
        });
}

Abstraction pair_only(const FlatGame& game) {
    return make_abstraction(game, "pair-only",
        [](const std::string& key) -> std::string {
            auto colon = key.find(':');
            if (colon == std::string::npos || colon <= 1)
                return key;  // preflop: no board card
            std::string out = key;
            out[1] = (key[0] == key[1]) ? 'P' : 'N';
            return out;
        });
}

} // namespace leduc_abs

} // namespace bluefish
