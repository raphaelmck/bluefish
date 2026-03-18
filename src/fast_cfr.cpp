// fast_cfr.cpp — High-performance CFR-family solvers.

#include "bluefish/fast_cfr.h"

#include <algorithm>
#include <string>

namespace bluefish {

// Fast vanilla CFR

FastCfrSolver::FastCfrSolver(FlatGame game, RootFn root_fn)
    : game_(std::move(game)), root_fn_(std::move(root_fn)) {
    table_.init(game_.num_info_sets, game_.is_num_actions);
}

void FastCfrSolver::ensure_synced() const {
    if (!synced_) {
        table_.to_info_map(
            const_cast<InfoMap&>(nodes_), game_.is_keys);
        const_cast<bool&>(synced_) = true;
    }
}

double FastCfrSolver::exploitability(RootFn /*root_fn*/) const {
    return flat_exploitability(game_, table_);
}

std::string FastCfrSolver::serialize_json() const {
    ensure_synced();
    return Solver::serialize_json();
}

const Solver::InfoMap& FastCfrSolver::info_map() const {
    ensure_synced();
    return nodes_;
}

double FastCfrSolver::train(int iterations, RootFn /*root_fn*/) {
    synced_ = false;
    double total = 0.0;
    for (int t = 0; t < iterations; ++t) {
        total += cfr(0, 1.0, 1.0);
        ++iterations_;
    }
    return total / static_cast<double>(iterations);
}

double FastCfrSolver::cfr(int ni, double pi0, double pi1) {
    auto& n = game_.nodes[static_cast<std::size_t>(ni)];
    ++stats_.nodes_touched;

    if (n.type == FlatNode::kTerminal) {
        ++stats_.terminal_visits;
        return n.utility_p0;
    }

    if (n.type == FlatNode::kChance) {
        ++stats_.chance_visits;
        double ev = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            double p = game_.chance_probs[ua];
            ev += p * cfr(game_.children[ua], pi0 * p, pi1 * p);
        }
        return ev;
    }

    ++stats_.info_set_visits;
    int is = n.info_set;
    int na = n.num_children;
    int off = table_.offset[static_cast<std::size_t>(is)];
    int fc = n.first_child;

    // Inline regret matching — no allocation
    double sigma[kMaxActions];
    double pos_sum = 0.0;
    for (int a = 0; a < na; ++a) {
        sigma[a] = std::max(0.0, table_.regret[static_cast<std::size_t>(off + a)]);
        pos_sum += sigma[a];
    }
    if (pos_sum > 0.0) {
        double inv = 1.0 / pos_sum;
        for (int a = 0; a < na; ++a) sigma[a] *= inv;
    } else {
        double u = 1.0 / static_cast<double>(na);
        for (int a = 0; a < na; ++a) sigma[a] = u;
    }

    double action_util[kMaxActions];
    double node_util = 0.0;

    for (int a = 0; a < na; ++a) {
        int child = game_.children[static_cast<std::size_t>(fc + a)];
        if (n.player == 0)
            action_util[a] = cfr(child, pi0 * sigma[a], pi1);
        else
            action_util[a] = cfr(child, pi0, pi1 * sigma[a]);
        node_util += sigma[a] * action_util[a];
    }

    for (int a = 0; a < na; ++a) {
        auto uoa = static_cast<std::size_t>(off + a);
        if (n.player == 0) {
            table_.regret[uoa]       += pi1 * (action_util[a] - node_util);
            table_.strategy_sum[uoa] += pi0 * sigma[a];
        } else {
            table_.regret[uoa]       += pi0 * (node_util - action_util[a]);
            table_.strategy_sum[uoa] += pi1 * sigma[a];
        }
    }
    return node_util;
}

// Fast CFR+

FastCfrPlusSolver::FastCfrPlusSolver(FlatGame game, RootFn root_fn)
    : game_(std::move(game)), root_fn_(std::move(root_fn)) {
    table_.init(game_.num_info_sets, game_.is_num_actions);
}

void FastCfrPlusSolver::ensure_synced() const {
    if (!synced_) {
        table_.to_info_map(
            const_cast<InfoMap&>(nodes_), game_.is_keys);
        const_cast<bool&>(synced_) = true;
    }
}

double FastCfrPlusSolver::exploitability(RootFn /*root_fn*/) const {
    return flat_exploitability(game_, table_);
}

std::string FastCfrPlusSolver::serialize_json() const {
    ensure_synced();
    return Solver::serialize_json();
}

const Solver::InfoMap& FastCfrPlusSolver::info_map() const {
    ensure_synced();
    return nodes_;
}

double FastCfrPlusSolver::train(int iterations, RootFn /*root_fn*/) {
    synced_ = false;
    double total = 0.0;
    for (int t = 0; t < iterations; ++t) {
        total += cfr_plus(0, 1.0, 1.0);
        ++iterations_;
    }
    return total / static_cast<double>(iterations);
}

double FastCfrPlusSolver::cfr_plus(int ni, double pi0, double pi1) {
    auto& n = game_.nodes[static_cast<std::size_t>(ni)];
    ++stats_.nodes_touched;

    if (n.type == FlatNode::kTerminal) {
        ++stats_.terminal_visits;
        return n.utility_p0;
    }

    if (n.type == FlatNode::kChance) {
        ++stats_.chance_visits;
        double ev = 0.0;
        int fc = n.first_child;
        for (int a = 0; a < n.num_children; ++a) {
            auto ua = static_cast<std::size_t>(fc + a);
            double p = game_.chance_probs[ua];
            ev += p * cfr_plus(game_.children[ua], pi0 * p, pi1 * p);
        }
        return ev;
    }

    ++stats_.info_set_visits;
    int is = n.info_set;
    int na = n.num_children;
    int off = table_.offset[static_cast<std::size_t>(is)];
    int fc = n.first_child;

    double sigma[kMaxActions];
    double pos_sum = 0.0;
    for (int a = 0; a < na; ++a) {
        sigma[a] = std::max(0.0, table_.regret[static_cast<std::size_t>(off + a)]);
        pos_sum += sigma[a];
    }
    if (pos_sum > 0.0) {
        double inv = 1.0 / pos_sum;
        for (int a = 0; a < na; ++a) sigma[a] *= inv;
    } else {
        double u = 1.0 / static_cast<double>(na);
        for (int a = 0; a < na; ++a) sigma[a] = u;
    }

    double action_util[kMaxActions];
    double node_util = 0.0;

    for (int a = 0; a < na; ++a) {
        int child = game_.children[static_cast<std::size_t>(fc + a)];
        if (n.player == 0)
            action_util[a] = cfr_plus(child, pi0 * sigma[a], pi1);
        else
            action_util[a] = cfr_plus(child, pi0, pi1 * sigma[a]);
        node_util += sigma[a] * action_util[a];
    }

    double t_weight = static_cast<double>(iterations_ + 1);

    for (int a = 0; a < na; ++a) {
        auto uoa = static_cast<std::size_t>(off + a);
        if (n.player == 0) {
            table_.regret[uoa] += pi1 * (action_util[a] - node_util);
            table_.strategy_sum[uoa] += t_weight * pi0 * sigma[a];
        } else {
            table_.regret[uoa] += pi0 * (node_util - action_util[a]);
            table_.strategy_sum[uoa] += t_weight * pi1 * sigma[a];
        }
        table_.regret[uoa] = std::max(0.0, table_.regret[uoa]);
    }
    return node_util;
}

std::string FastCfrPlusSolver::validate() const {
    // Check base invariants via lazy sync
    ensure_synced();
    std::string base_err = Solver::validate();
    if (!base_err.empty()) return base_err;
    // CFR+: all regrets must be non-negative
    for (int is = 0; is < game_.num_info_sets; ++is) {
        int off = table_.offset[static_cast<std::size_t>(is)];
        int na = table_.num_actions[static_cast<std::size_t>(is)];
        for (int a = 0; a < na; ++a) {
            if (table_.regret[static_cast<std::size_t>(off + a)] < -1e-12)
                return "fast-cfr+ invariant: info set '" +
                       game_.is_keys[static_cast<std::size_t>(is)] +
                       "' has negative regret";
        }
    }
    return {};
}

// Fast external-sampling MCCFR

FastMccfrSolver::FastMccfrSolver(FlatGame game, RootFn root_fn,
                                  uint64_t seed)
    : game_(std::move(game)), root_fn_(std::move(root_fn)), rng_(seed) {
    table_.init(game_.num_info_sets, game_.is_num_actions);
}

void FastMccfrSolver::ensure_synced() const {
    if (!synced_) {
        table_.to_info_map(
            const_cast<InfoMap&>(nodes_), game_.is_keys);
        const_cast<bool&>(synced_) = true;
    }
}

double FastMccfrSolver::exploitability(RootFn /*root_fn*/) const {
    return flat_exploitability(game_, table_);
}

std::string FastMccfrSolver::serialize_json() const {
    ensure_synced();
    return Solver::serialize_json();
}

const Solver::InfoMap& FastMccfrSolver::info_map() const {
    ensure_synced();
    return nodes_;
}

int FastMccfrSolver::sample(const double* probs, int n) {
    double r = dist_(rng_);
    double cum = 0.0;
    for (int a = 0; a < n; ++a) {
        cum += probs[a];
        if (r < cum) return a;
    }
    return n - 1;
}

double FastMccfrSolver::train(int iterations, RootFn /*root_fn*/) {
    synced_ = false;
    double total = 0.0;
    for (int t = 0; t < iterations; ++t) {
        int traverser = static_cast<int>(iterations_ % 2);
        double v = traverse(0, traverser);
        total += (traverser == 0) ? v : -v;
        ++iterations_;
    }
    return total / static_cast<double>(iterations);
}

double FastMccfrSolver::traverse(int ni, int traverser) {
    auto& n = game_.nodes[static_cast<std::size_t>(ni)];
    ++stats_.nodes_touched;

    if (n.type == FlatNode::kTerminal) {
        ++stats_.terminal_visits;
        return (traverser == 0) ? n.utility_p0 : -n.utility_p0;
    }

    if (n.type == FlatNode::kChance) {
        ++stats_.chance_visits;
        int fc = n.first_child;
        int a = sample(&game_.chance_probs[static_cast<std::size_t>(fc)],
                        n.num_children);
        return traverse(game_.children[static_cast<std::size_t>(fc + a)],
                        traverser);
    }

    ++stats_.info_set_visits;
    int is = n.info_set;
    int na = n.num_children;
    int off = table_.offset[static_cast<std::size_t>(is)];
    int fc = n.first_child;

    double sigma[kMaxActions];
    double pos_sum = 0.0;
    for (int a = 0; a < na; ++a) {
        sigma[a] = std::max(0.0, table_.regret[static_cast<std::size_t>(off + a)]);
        pos_sum += sigma[a];
    }
    if (pos_sum > 0.0) {
        double inv = 1.0 / pos_sum;
        for (int a = 0; a < na; ++a) sigma[a] *= inv;
    } else {
        double u = 1.0 / static_cast<double>(na);
        for (int a = 0; a < na; ++a) sigma[a] = u;
    }

    if (n.player == traverser) {
        double action_util[kMaxActions];
        double node_util = 0.0;

        for (int a = 0; a < na; ++a) {
            int child = game_.children[static_cast<std::size_t>(fc + a)];
            action_util[a] = traverse(child, traverser);
            node_util += sigma[a] * action_util[a];
        }

        for (int a = 0; a < na; ++a) {
            table_.regret[static_cast<std::size_t>(off + a)] +=
                action_util[a] - node_util;
        }
        return node_util;
    }

    for (int a = 0; a < na; ++a) {
        table_.strategy_sum[static_cast<std::size_t>(off + a)] += sigma[a];
    }

    int sampled = sample(sigma, na);
    int child = game_.children[static_cast<std::size_t>(fc + sampled)];
    return traverse(child, traverser);
}

} // namespace bluefish
