// bluefish/solver.h - Shared solver infrastructure

#pragma once

#include "game.h"

#include <functional>

namespace bluefish {

// - Information set node -

struct InfoNode {
    std::vector<double> regret_sum;
    std::vector<double> strategy_sum;
    int num_actions = 0;

    void init(int n);

    std::vector<double> current_strategy() const;

    // Time-averaged strategy — the convergent Nash output.
    std::vector<double> average_strategy() const;
};

// - Traversal statistics -

struct Stats {
    int64_t nodes_touched   = 0;  // every recursive call (decision + terminal + chance)
    int64_t terminal_visits = 0;  // leaf nodes reached
    int64_t info_set_visits = 0;  // decision nodes (info set lookups)
    int64_t chance_visits   = 0;  // chance nodes traversed or sampled

    void reset() {
        nodes_touched = terminal_visits = info_set_visits = chance_visits = 0;
    }
};

// - Solver base class -

class Solver {
public:
    using RootFn  = std::function<std::vector<RootState>()>;
    using InfoMap = std::unordered_map<std::string, InfoNode>;

    virtual ~Solver() = default;

    virtual double train(int iterations, RootFn root_fn) = 0;

    virtual std::string name() const = 0;

    // - Shared evaluation tools -

    virtual double exploitability(RootFn root_fn) const;

    virtual std::string serialize_json() const;

    // - Validation -

    virtual std::string validate() const;

    // - Accessors -

    virtual const InfoMap& info_map() const { return nodes_; }
    virtual std::size_t num_info_sets() const { return nodes_.size(); }
    int64_t iterations() const { return iterations_; }
    const Stats& stats() const { return stats_; }

protected:
    mutable InfoMap nodes_;  // mutable: fast solvers populate lazily
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
