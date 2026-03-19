// parallel_mccfr.cpp — Lock-free parallel external-sampling MCCFR.

#include "bluefish/parallel_mccfr.h"

#include <random>
#include <thread>

namespace bluefish {

namespace {

inline constexpr std::size_t kCacheLine = 64;

struct alignas(kCacheLine) Worker {
    const FlatGame* game;
    // Thread-local regret/strategy arrays. Same layout as shared table
    std::vector<double> local_regret;
    std::vector<double> local_strat_sum;
    const std::vector<int32_t>* offsets;
    const std::vector<int32_t>* num_act;

    std::mt19937_64 rng;
    std::uniform_real_distribution<double> dist{0.0, 1.0};
    Stats stats{};

    Worker() : game(nullptr), offsets(nullptr), num_act(nullptr), rng(0) {}

    void init(const FlatGame& g, const RegretTable& shared, uint64_t seed) {
        game = &g;
        offsets = &shared.offset;
        num_act = &shared.num_actions;
        // Copy current regret state so regret matching uses latest strategy
        local_regret = shared.regret;
        // Start strategy sum at zero — we accumulate deltas
        local_strat_sum.assign(shared.strategy_sum.size(), 0.0);
        rng.seed(seed);
        stats = {};
    }

    int sample(const double* probs, int n) {
        double r = dist(rng);
        double cum = 0.0;
        for (int a = 0; a < n; ++a) {
            cum += probs[a];
            if (r < cum) return a;
        }
        return n - 1;
    }

    double traverse(int ni, int traverser) {
        auto& nd = game->nodes[static_cast<std::size_t>(ni)];
        ++stats.nodes_touched;

        if (nd.type == FlatNode::kTerminal) {
            ++stats.terminal_visits;
            return (traverser == 0) ? nd.utility_p0 : -nd.utility_p0;
        }

        if (nd.type == FlatNode::kChance) {
            ++stats.chance_visits;
            int fc = nd.first_child;
            int a = sample(
                &game->chance_probs[static_cast<std::size_t>(fc)],
                nd.num_children);
            return traverse(
                game->children[static_cast<std::size_t>(fc + a)],
                traverser);
        }

        ++stats.info_set_visits;
        int is = nd.info_set;
        int na = nd.num_children;
        int off = (*offsets)[static_cast<std::size_t>(is)];
        int fc = nd.first_child;

        // Regret matching from thread-local regrets
        double sigma[kMaxActions];
        double pos_sum = 0.0;
        for (int a = 0; a < na; ++a) {
            sigma[a] = std::max(0.0,
                local_regret[static_cast<std::size_t>(off + a)]);
            pos_sum += sigma[a];
        }
        if (pos_sum > 0.0) {
            double inv = 1.0 / pos_sum;
            for (int a = 0; a < na; ++a) sigma[a] *= inv;
        } else {
            double u = 1.0 / static_cast<double>(na);
            for (int a = 0; a < na; ++a) sigma[a] = u;
        }

        if (nd.player == traverser) {
            double action_util[kMaxActions];
            double node_util = 0.0;
            for (int a = 0; a < na; ++a) {
                int child = game->children[
                    static_cast<std::size_t>(fc + a)];
                action_util[a] = traverse(child, traverser);
                node_util += sigma[a] * action_util[a];
            }
            // Write to thread-local regrets
            for (int a = 0; a < na; ++a) {
                local_regret[static_cast<std::size_t>(off + a)] +=
                    action_util[a] - node_util;
            }
            return node_util;
        }

        // Opponent: accumulate into thread-local strategy sum
        for (int a = 0; a < na; ++a) {
            local_strat_sum[static_cast<std::size_t>(off + a)] += sigma[a];
        }
        int sampled = sample(sigma, na);
        int child = game->children[
            static_cast<std::size_t>(fc + sampled)];
        return traverse(child, traverser);
    }

    void run(int iterations, int64_t start_iter) {
        for (int t = 0; t < iterations; ++t) {
            int traverser = static_cast<int>((start_iter + t) % 2);
            traverse(0, traverser);
        }
    }
};

void thread_entry(Worker* w, int iterations, int64_t start_iter) {
    w->run(iterations, start_iter);
}

} // namespace

// - ParallelMccfrSolver -

ParallelMccfrSolver::ParallelMccfrSolver(
    FlatGame game, RootFn root_fn, int num_threads, uint64_t seed)
    : game_(std::move(game))
    , root_fn_(std::move(root_fn))
    , num_threads_(num_threads > 0
          ? num_threads
          : std::max(1, static_cast<int>(
                std::thread::hardware_concurrency())))
    , base_seed_(seed)
{
    table_.init(game_.num_info_sets, game_.is_num_actions);
}

std::string ParallelMccfrSolver::name() const {
    return "par-mccfr-" + std::to_string(num_threads_) + "t";
}

void ParallelMccfrSolver::ensure_synced() const {
    if (!synced_) {
        table_.to_info_map(
            const_cast<InfoMap&>(nodes_), game_.is_keys);
        const_cast<bool&>(synced_) = true;
    }
}

double ParallelMccfrSolver::exploitability(RootFn /*root_fn*/) const {
    return flat_exploitability(game_, table_);
}

std::string ParallelMccfrSolver::serialize_json() const {
    ensure_synced();
    return Solver::serialize_json();
}

const Solver::InfoMap& ParallelMccfrSolver::info_map() const {
    ensure_synced();
    return nodes_;
}

double ParallelMccfrSolver::train(int iterations, RootFn /*root_fn*/) {
    synced_ = false;

    int per_thread = iterations / num_threads_;
    int remainder = iterations % num_threads_;

    // Allocate per-thread workers with their own regret copies
    std::vector<std::unique_ptr<Worker>> workers;
    workers.reserve(static_cast<std::size_t>(num_threads_));
    for (int i = 0; i < num_threads_; ++i) {
        auto w = std::make_unique<Worker>();
        uint64_t seed = base_seed_
            + static_cast<uint64_t>(iterations_) * 997
            + static_cast<uint64_t>(i) * 7919;
        w->init(game_, table_, seed);
        workers.push_back(std::move(w));
    }

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads_));
    int64_t iter_offset = iterations_;

    for (int i = 0; i < num_threads_; ++i) {
        int batch = per_thread + (i < remainder ? 1 : 0);
        Worker* wp = workers[static_cast<std::size_t>(i)].get();
        threads.emplace_back(thread_entry, wp, batch, iter_offset);
        iter_offset += batch;
    }

    for (auto& t : threads)
        t.join();

    auto n_entries = static_cast<std::size_t>(table_.total_entries);
    
    double inv_n = 1.0 / static_cast<double>(num_threads_);
    for (std::size_t j = 0; j < n_entries; ++j) {
        double sum_r = 0.0;
        double sum_s = 0.0;
        for (auto& w : workers) {
            sum_r += w->local_regret[j];
            sum_s += w->local_strat_sum[j];
        }
        table_.regret[j] = sum_r * inv_n;
        table_.strategy_sum[j] += sum_s;
    }

    for (auto& w : workers) {
        stats_.nodes_touched   += w->stats.nodes_touched;
        stats_.terminal_visits += w->stats.terminal_visits;
        stats_.info_set_visits += w->stats.info_set_visits;
        stats_.chance_visits   += w->stats.chance_visits;
    }

    iterations_ += iterations;
    return 0.0;
}

} // namespace bluefish
